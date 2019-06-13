#!/usr/bin/env python
# -*- coding: utf-8 -*-

#############################################################################
"""

#############################################################################

Copyright (c) 2018 - Nathaniel Starkman
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  The name of the author may not be used to endorse or promote products
     derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

#############################################################################
"""

#############################################################################
# Imports

from .Orbits import Orbit

#############################################################################
# Info

__author__ = "Nathaniel Starkman"
__copyright__ = "Copyright 2018, "
__credits__ = ["Jo Bovy"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Nathaniel Starkman"
__email__ = "n.starkman@mail.utoronto.ca"
__status__ = "Production"

###############################################################################
# Code


class NSOrbit(Orbit):
    """docstring for NSOrbits"""

    def _call_internal(self, *args, **kwargs):
        """
        NAME:
            _call_internal
        PURPOSE:
            return the orbits vector at time t (like OrbitTop's __call__)
        INPUT:
            t - desired time
                besides standard galpy inputs, supports:
                    'full' -> self.t
                    'start' -> self.t[0]
                    'end' -> self.t[-1]
        OUTPUT:
            [R, vR, vT, z, vz(, phi)] or [R, vR, vT(, phi)] depending on the orbit;
        HISTORY:
            2019-03-01 - Written - Nathaniel Starkman (UofT)
        """
        # redirect blank args back to standard Orbits
        if not args:
            return super()._call_internal(*args, **kwargs)

        # adding string options
        elif isinstance(args[0], str):
            if args[0] == 'full':  # shortcut for t = self.t
                return self.orbit.T
            elif args[0] == 'start':
                return self.orbit[0].T
            if args[0] == 'end':
                return self.orbit[-1].T
            else:
                raise ValueError("time is not in ('full', 'start', 'end')")

        # redirect all other options back to standard Orbits
        else:
            return super()._call_internal(*args, **kwargs)
    # /def

    def __eq__(self, other):
        """basic test of equality"""
        return (self.getOrbit() == other.getOrbit()).all()
    # /def
