<p align="center">
    <a href="http://www.galpy.org" target="_blank"><img src="https://galpy.readthedocs.io/en/latest/_static/galpy-logo-small.gif"></a><br/>
    <b>Galactic Dynamics in python</b>
</p>

[galpy](http://www.galpy.org) is a Python 2 and 3 package for galactic dynamics. It supports orbit integration in a variety of potentials, evaluating and sampling various distribution functions, and the calculation of action-angle coordinates for all static potentials. `galpy` is an [astropy](http://www.astropy.org/) [affiliated package](http://www.astropy.org/affiliated/) and provides full support for astropy’s [Quantity](http://docs.astropy.org/en/stable/api/astropy.units.Quantity.html) framework for variables with units.

[![image](https://travis-ci.org/jobovy/galpy.svg?branch=master)](http://travis-ci.org/jobovy/galpy) [![image](https://ci.appveyor.com/api/projects/status/wmgs1sq3i7tbtap2/branch/master?svg=true)](https://ci.appveyor.com/project/jobovy/galpy) [![image](http://codecov.io/github/jobovy/galpy/coverage.svg?branch=master)](http://codecov.io/github/jobovy/galpy?branch=master) [![image](https://readthedocs.org/projects/galpy/badge/?version=latest)](http://docs.galpy.org/en/latest/) [![image](http://img.shields.io/pypi/v/galpy.svg)](https://pypi.python.org/pypi/galpy/) [![image](https://anaconda.org/conda-forge/galpy/badges/installer/conda.svg)](https://anaconda.org/conda-forge/galpy) [![image](http://img.shields.io/badge/license-New%20BSD-brightgreen.svg)](https://github.com/jobovy/galpy/blob/master/LICENSE) [![image](http://img.shields.io/badge/DOI-10.1088/0067%2D%2D0049/216/2/29-blue.svg)](http://dx.doi.org/10.1088/0067-0049/216/2/29) [![image](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/) [![image](https://slackin-galpy.herokuapp.com/badge.svg)](https://galpy.slack.com/) [![image](https://img.shields.io/badge/join-slack-E01563.svg?style=flat&logo=slack&logoWidth=10)](https://slackin-galpy.herokuapp.com)

AUTHOR
======

Jo Bovy - bovy at astro dot utoronto dot ca

See
[AUTHORS.txt](https://github.com/jobovy/galpy/blob/master/AUTHORS.txt)
for a full list of contributors.

If you find this code useful in your research, please let me know. **If
you use galpy in a publication, please cite** [Bovy
(2015)](http://adsabs.harvard.edu/abs/2015ApJS..216...29B) **and link to
http://github.com/jobovy/galpy**. See [the acknowledgement documentation
section](http://docs.galpy.org/en/latest/index.html#acknowledging-galpy)
for a more detailed guide to citing parts of the code. Thanks!

LOOKING FOR HELP?
=================

The latest documentation can be found
[here](http://docs.galpy.org/en/latest/). You can also join the
[galpy slack community](https://galpy.slack.com/) for any questions
related to `galpy`; join
[here](https://slackin-galpy.herokuapp.com).

If you find *any* bug in the code, please report these using the [Issue
Tracker](http://github.com/jobovy/galpy/issues) or by joining the [galpy
slack community](https://galpy.slack.com/).

If you are having issues with the installation of `galpy`, please first
consult the [Installation
FAQ](http://docs.galpy.org/en/latest/installation.html#installation-faq).

PYTHON VERSIONS AND DEPENDENCIES
================================

`galpy` supports both Python 2 and 3. Specifically, galpy supports
Python 2.7 and Python 3.7, 3.8, 3.9. It should also work on earlier
Python 3.\* versions, but this is not extensively tested on an ongoing
basis.  Travis CI builds regularly check support for Python 3.9 (and
of 2.7, 3.7, and 3.8 using a more limited, core set of tests) and
Appveyor builds regularly check support for Python 3.9 on Windows.

This package requires [Numpy](https://numpy.org/),
[Scipy](http://www.scipy.org/), and
[Matplotlib](http://matplotlib.sourceforge.net/). Certain advanced
features require the GNU Scientific Library
([GSL](http://www.gnu.org/software/gsl/)), with action calculations
requiring version 1.14 or higher. Use of `SnapshotRZPotential` and
`InterpSnapshotRZPotential` requires
[pynbody](https://github.com/pynbody/pynbody). Support for providing
inputs and getting outputs as Quantities with units is provided through
[astropy](http://www.astropy.org/). Certain parts of the code require 
additional packages and you will be alerted by the code if they are
not installed.

CONTRIBUTING TO GALPY
=====================

If you are interested in contributing to galpy\'s development, take a
look at [this brief
guide](https://github.com/jobovy/galpy/wiki/Guide-for-new-contributors)
on the wiki. This will hopefully help you get started!

Some further development notes can be found on the
[wiki](http://github.com/jobovy/galpy/wiki/). This includes a list of
small and larger extensions of galpy that would be useful
[here](http://github.com/jobovy/galpy/wiki/Possible-galpy-extensions) as
well as a longer-term roadmap
[here](http://github.com/jobovy/galpy/wiki/Roadmap). Please let the main
developer know if you need any help contributing!

DISK DF CORRECTIONS
===================

The dehnendf and shudf disk distribution functions can be corrected to
follow the desired surface-mass density and radial-velocity-dispersion
profiles more closely (see
[1999AJ\....118.1201D](http://adsabs.harvard.edu/abs/1999AJ....118.1201D)).
Calculating these corrections is expensive, and a large set of
precalculated corrections can be found
[here](http://github.com/downloads/jobovy/galpy/galpy-dfcorrections.tar.gz)
\[tar.gz archive\]. Install these by downloading them and unpacking them
into the galpy/df/data directory before running the setup.py
installation. E.g.:

    curl -O https://github.s3.amazonaws.com/downloads/jobovy/galpy/galpy-dfcorrections.tar.gz
    tar xvzf galpy-dfcorrections.tar.gz -C ./galpy/df/data/
