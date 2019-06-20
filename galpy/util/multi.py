#Brian Refsdal's parallel_map, from astropython.org
#Not sure what license this is released under, but until I know better:
#
#Copyright (c) 2010, Brian Refsdal
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are
#met: 
#
#1. Redistributions of source code must retain the above copyright
#notice, this list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright
#notice, this list of conditions and the following disclaimer in the
#documentation and/or other materials provided with the distribution.
#
#3. The name of the author may not be used to endorse or promote
#products derived from this software without specific prior written
#permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#############################################################################
# IMPORTS

from __future__ import print_function
import platform
import numpy as np
import time

try:
    # May raise ImportError
    import multiprocessing

except ImportError as e:
    _multi = False
else:
    _multi = True

try:
    # May raise NotImplementedError
    _ncpus = multiprocessing.cpu_count()
except NotImplementedError as e:
    _ncpus = 1

# a progressbar
try:
    from tqdm import tqdm as TQDM
except ImportError:
    TQDM = lambda x, *args, **kw: x  # blanck TQDM

#############################################################################
# INFO

__all__ = ('parallel_map',)

#############################################################################
# CODE


def worker(func, ii, chunk, out_q, err_q, lock, tqdm=TQDM):
    """
    A worker function that maps an input function over a
    slice of the input iterable.

    :param func  : callable function that accepts argument from iterable
    :param ii  : process ID
    :param chunk: slice of input iterable
    :param out_q: thread-safe output queue
    :param err_q: thread-safe queue to populate on exception
    :param lock : thread-safe lock to protect a resource
         ( useful in extending parallel_map() )
    :param tqdm : progressbar function, internally managed
    """
    # initializing
    vals = []
    sec = '\tWorker {}: len={}'.format(ii, len(chunk))

    # iterate over slice
    for val in tqdm(chunk, leave=True, desc=sec):
        try:
            result = func(val)
        except Exception as e:
            err_q.put(e)
            return

        vals.append(result)

    # output the result and task ID to output queue
    out_q.put((ii, vals))
# /def


def run_tasks(procs, err_q, out_q, num, size, tqdm=TQDM):
    """
    A function that executes populated processes and processes
    the resultant array. Checks error queue for any exceptions.

    :param procs: list of Process objects
    :param out_q: thread-safe output queue
    :param err_q: thread-safe queue to populate on exception
    :param num  : length of resultant array
    :param size : size of iterable
    :param tqdm : progressbar function, internally managed
    :param tqdm : progressbar function, internally managed
    """
    # function to terminate processes that are still running.
    die = (lambda vs: [v.terminate() for v in vs if v.exitcode is None])

    try:
        for proc in procs:  # starting processes
            proc.start()

        for proc in procs:  # ending the processes
            proc.join()

    except Exception as e:
        # kill all slave processes on ctrl-C
        try:
            die(procs)
        finally:
            raise e

    else:
        pass

    if not err_q.empty():
        # kill all on any exception from any one slave
        try:
            die(procs)
        finally:

            raise err_q.get()

    # Processes finish in arbitrary order. Process IDs double
    # as index in the resultant array.)
    indices = np.array_split(np.arange(size), num)
    results = np.empty(size, dtype=list)
    # assigning values
    while not out_q.empty():
        idx, result = out_q.get()
        results[indices[idx]] = result

    return results
# /def


def parallel_map(function, sequence, func_args=[], func_kws={},
                 numcores=None, _progressbar=False):
    """
    A parallelized version of the native Python map function that
    utilizes the Python multiprocessing module to divide and
    conquer sequence.

    expanded to allow extra arguments to the function through:
      func_args & func_kws

    :param function : callable function that accepts argument from iterable
    :param sequence : iterable sequence
    :param func_args: extra function arguments. not iterated over.
    :param func_kws : extra function kwargs. not iterated over.
    :param numcores : number of cores to use
    :param _progressbar : whether to display the progressbar
    """
    if not callable(function):
        raise TypeError(f"input function {function} is not callable")

    if not np.iterable(sequence):
        raise TypeError(f"input {sequence} is not iterable")

    if _progressbar:
        tqdm = TQDM
    else:
        tqdm = lambda x, *args, **kw: x

    size = len(sequence)

    func = lambda x: function(x, *func_args, **func_kws)  # @ NS

    if not _multi or size == 1:
        return map(func, sequence)

    if numcores is None:
        numcores = _ncpus

    if platform.system() == 'Windows': # JB: don't think this works on Win
        return list(map(func, sequence))

    # Returns a started SyncManager object which can be used for sharing
    # objects between processes. The returned manager object corresponds
    # to a spawned child process and has methods which will create shared
    # objects and return corresponding proxies.
    manager = multiprocessing.Manager()

    # Create FIFO queue and lock shared objects and return proxies to them.
    # The managers handles a server process that manages shared objects that
    # each slave process has access to. Bottom line -- thread-safe.
    out_q = manager.Queue()
    err_q = manager.Queue()
    lock = manager.Lock()

    # if sequence is less than numcores, only use len sequence number of
    # processes
    if size < numcores:
        numcores = size

    # group sequence into numcores-worth of chunks
    sequence = np.array_split(sequence, numcores)

    procs = [multiprocessing.Process(
                target=worker,
                args=(func, ii, chunk, out_q, err_q, lock, tqdm))
             for ii, chunk in enumerate(tqdm(sequence, desc='Master Loop'))]

    return run_tasks(procs, err_q, out_q, numcores, size, tqdm=tqdm)
# /def


if __name__ == "__main__":
  """
  Unit test of parallel_map()

  Create an arbitrary length list of references to a single
  matrix containing random floats and compute the eigenvals
  in serial and parallel. Compare the results and timings.
  """

  import time

  numtasks = 5
  #size = (1024,1024)
  size = (512,512)

  vals = numpy.random.rand(*size)
  f = numpy.linalg.eigvals

  iterable = [vals]*numtasks

  print ('Running numpy.linalg.eigvals %iX on matrix size [%i,%i]' %
      (numtasks,size[0],size[1]))

  tt = time.time()
  presult = parallel_map(f, iterable)
  print('parallel map in %g secs' % (time.time()-tt))

  tt = time.time()
  result = map(f, iterable)
  print('serial map in %g secs' % (time.time()-tt))

  assert (numpy.asarray(result) == numpy.asarray(presult)).all()


