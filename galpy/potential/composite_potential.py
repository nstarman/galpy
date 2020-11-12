# -*- coding: utf-8 -*-

"""Composite Potential."""

__all__ = ["CompositePotential"]

##############################################################################
# IMPORTS

# BUILT-IN
import copy
import typing as T
from collections import OrderedDict

# THIRD PARTY
import numpy as np

# PROJECT-SPECIFIC
from .Potential import Potential
from galpy.util.config import __config__ as config


##############################################################################
# TYPES

GalpyPotentialType = T.TypeVar(
    "GalpyPotentialType", Potential, T.List[Potential]
)

##############################################################################
# CODE
##############################################################################


def indent(s: str, shift: int = 1, width: int = 4) -> str:
    """Indent a block of text.  The indentation is applied to each line.

    Function from Astropy.

    Parameters
    ----------
    s : str
    shift : int
    width : int

    Returns
    -------
    str

    """
    indented = "\n".join(
        " " * (width * shift) + (line if line else "")
        for line in s.splitlines()
    )
    if s[-1] == "\n":
        indented += "\n"

    return indented


# -------------------------------------------------------------------


class CompositePotential(Potential, OrderedDict):
    """A composite Galpy potential composed of distinct components.

    In ``galpy``, composite potentials are represented as lists. This is a
    very pythonic implementation as it allows for fast iteration, maintaining
    order, and nesting. One drawback is that the components cannot be named.
    We introduce a compatibility class for ``Potential``s that is both a list
    and also key-accessed by component name.

    Parameters
    ----------
    *args
        list
    **kwargs

    Examples
    --------
    As an example, here is a recreation of ``MilkyWayPotential2014``.

        >>> from galpy import potential as gpot
        >>> pot = CompositePotential(
        ...     disk=gpot.PowerSphericalPotentialwCutoff(alpha=1.8, rc=1.9/8,
        ...                                              normalize=0.05),
        ...     disc=gpot.MiyamotoNagaiPotential(a=3./8,b=0.28/8,normalize=.6),
        ...     halo=gpot.NFWPotential(a=16/8., normalize=.35))
        >>> print(pot)  # doctest: +ELLIPSIS
        CompositePotential:
          disk : PowerSphericalPotentialwCutoff (at ...)
          disc : MiyamotoNagaiPotential (at ...)
          halo : NFWPotential (at ...)

    This potential, while technically an ordered dictionary, overrides the
    ``__class__`` attribute and ``__iter__`` methods to mimic a list.

        >>> isinstance(pot, list)
        True

    And looks exactly like the built-in option.

        >>> pot  # doctest: skip
        [PowerSphericalPotentialwCutoff, MiyamotoNagaiPotential, NFWPotential]

        >>> galpot.MWPotential2014  # doctest: skip
        [PowerSphericalPotentialwCutoff, MiyamotoNagaiPotential, NFWPotential]

    Consequently, it functions exactly as a stand-in.


    .. plot::
        :include-source:

        import galpy.potential as gpot
        from galpy.orbit import Orbit
        import matplotlib.pyplot as plt
        import numpy as np

        ts = np.linspace(0, 1, num=50)
        o1 = Orbit()  # Potential
        o1.integrate(ts, pot)

        o2 = Orbit()  # galpy built-in
        o2.integrate(ts, galpot.MWPotential2014)

        # plot
        o1.plot()
        o2.plot(ls="--", overplot=True)
        plt.legend()
        plt.show()

    Similarly to how ``galpy`` can extend lists of potentials, dictionary
    operations may be used for extending ``CompositePotential`` objects.

    We use the same example as in the `docs
    <https://docs.galpy.org/en/v1.6.0/reference/potential.html>`_:

    "If one wants to add the supermassive black hole at the Galactic center,
    this can be done by":

        >>> potwBH = pot + {"BH": galpot.KeplerPotential(amp=4e6 * u.solMass)}
        >>> print(potwBH)  # doctest: +ELLIPSIS
        CompositePotential:
          disk : PowerSphericalPotentialwCutoff (at ...)
          disc : MiyamotoNagaiPotential (at ...)
          halo : NFWPotential (at ...)
          BH : KeplerPotential (at ...)

    Composite potentials may also be added:

        >>> potwBH = pot + CompositePotential(
        ...     BH=galpot.KeplerPotential(amp=4e6 * u.solMass))
        >>> print(potwBH)
        CompositePotential:
          disk : PowerSphericalPotentialwCutoff (at ...)
          disc : MiyamotoNagaiPotential (at ...)
          halo : NFWPotential (at ...)
          BH : KeplerPotential (at ...)

    """

    def __init__(self, *args, **kwargs):

        self._dim: int = None
        self._isRZ: bool = None
        self._isNonAxi: bool = None
        # self._hasC: bool = None
        # self._hasC_dxdv: bool = None
        # self._hasC_dens: bool = None

        self._amp = 1.0
        self._ro: float = None
        self._vo: float = None

        if len(args) > 0 and isinstance(args[0], list):
            for k, v in args[0]:
                kwargs[k] = v
        else:
            for i, v in args:
                kwargs[str(i)] = v

        OrderedDict.__init__(self, **kwargs)

    def _check_potential(self, p):
        if not isinstance(p, Potential):
            raise TypeError(
                "Potential components may only be Potential "
                f"objects, not {type(p)}."
            )

        if self._dim is None:  # only the first time
            self._dim = p.dim
            self._isRZ = p.isRZ
            self._isNonAxi = p.isNonAxi
            # self._hasC = p.hasC
            # self._hasC_dxdv = p.hasC_dxdv
            # self._hasC_dens = p.hasC_dens
            self._ro = p._ro
            self._vo = p._vo
        elif p.dim != self.dim:
            raise ValueError(
                "All potentials must have the same number "
                f"of dimensions ({self[0]} has {self.dim})."
            )
        elif (p._ro != self._ro) or (p._vo != self._vo):
            raise ValueError(
                "All potentials must have the same ro, vo "
                f"({self[0]} has {self._ro}, {self._vo})."
            )

    # ----------------------

    def __getitem__(self, key: T.Union[str, int]):
        """Get item, checking component first."""
        if isinstance(key, int):
            key = tuple(self.keys())[key]

        return super().__getitem__(key)

    def __setitem__(self, key, value):
        """Set item, checking component first."""
        self._check_potential(value)
        super().__setitem__(key, value)

    def __iter__(self):
        """Iterate like a list."""
        return iter(self.values())

    @property
    def __class__(self):
        """Fake class for working with a `list`."""
        return list

    def __repr__(self):
        """String representation.

        Returns
        -------
        str

        """
        rep: str = f"CompositePotential: (at <{hex(id(self))}>)\n"

        for key, pot in self.items():
            if isinstance(pot, CompositePotential):
                pot_name = pot.__class__.__name__
                s = indent(f"{key} : {repr(pot)}", width=2)

            else:  # a normal string
                pot_name = pot.__class__.__name__
                s = indent(f"{key} : {pot_name}\n", width=2)

            rep += s

        return rep

    # ----------------------

    def __add__(self, other):
        """Add together using ``merge_strategy``, copying self and other."""
        return merge(
            copy.deepcopy(self), copy.deepcopy(other), merge_func=None,
        )

    def __iadd__(self, other):
        """Add in-place using ``merge_strategy``, copying self and other."""
        return merge(self, other, merge_func=None)

    def __radd__(self, other):
        """Add in-place using ``merge_strategy``, copying self and other."""
        return self.__add__(other)

    def __mul__(self, other: float):
        """Multiply each component by ``other``."""
        other = float(other)
        cpot = copy.deepcopy(self)

        for k in cpot.keys():
            cpot[k] = cpot[k] * other

        return cpot

    def __imul__(self, other):
        """In-place multiply each component by ``other``."""
        other = float(other)
        for k in self.keys():
            self[k] = self[k] * other
        return self

    def __rmul__(self, other):
        """Multiply each component by ``other``."""
        return self.__mul__(other)

    # ----------------------

    @property
    def dim(self):
        """Dimension of potential."""
        return self._dim

    @property
    def isRZ(self):
        """Whether the potential is RZ."""
        return self._isRZ

    @property
    def isNonAxi(self):
        """Whether the potential is non-axisymmetric."""
        return self._isNonAxi

    @property
    def hasC(self):
        """Whether the potential has a C implementation."""
        return False

    @property
    def hasC_dxdv(self):
        """Whether the potential has a C phase-space implementation."""
        return False

    @property
    def hasC_dens(self):
        """Whether the potential has a C density."""
        return False

    # ----------------------

    __deepcopy__ = None

    def __getattr__(self, attr):
        """Pure Magic."""
        return lambda *args, **kwargs: np.sum(
            [getattr(p, attr)(*args, **kwargs) for p in self]
        )


# /class


# ----------------------------------------------------------------------------


def merge(left, right, merge_func=None):
    """Merge the ``left`` and ``right`` potential objects.

    .. todo::

        Merge strategy
        https://docs.astropy.org/en/stable/utils/index.html

    Parameters
    ----------
    left
    right

    merge_func : str
        One of 'overwrite', 'rename', 'merge'

    Returns
    -------
    merged

    """
    if merge_func is None:
        merge_func = config.get("potential", "merge")

    # -------------

    if merge_func == "overwrite":
        pass

    elif merge_func == "rename":
        right_keys = tuple(right.keys())
        for k in right_keys:
            if k in left:
                left[k + "_0"] = left.pop(k)
                right[k + "_1"] = right.pop(k)

    elif merge_func == "merge":
        right_keys = tuple(right.keys())
        for k in right_keys:
            if k in left:
                left[k] = CompositePotential(
                    **{"comp_0": left[k], "comp_1": right.pop(k)}
                )

    else:
        raise ValueError("`merge_strategy` not right")

    # -------------

    left.update(right)
    return left


# /def


##############################################################################
# END
