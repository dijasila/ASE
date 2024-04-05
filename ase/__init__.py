# Copyright 2008, 2009 CAMd
# (see accompanying license files for details).

"""Atomic Simulation Environment."""

# import ase.parallel early to avoid circular import problems when
# ase.parallel does "from gpaw.mpi import world":
import ase.parallel  # noqa
from ase.atom import Atom
from ase.atoms import Atoms

__all__ = ['Atoms', 'Atom']
__version__ = '3.23.0b1'    # Change also in project.toml

ase.parallel  # silence pyflakes

"""
This is a kind of magic, described here: `Packaging namespace packages
<https://packaging.python.org/en/latest/guides/packaging-namespace-packages/>`
We use a pkgutil-style namespace () package here, to allow use __init__.py.
However, the plugins writers should use PEP420 native namespace package
approach: they definitelly should NOT include __init__.py in ase and
ase.plugins package.
So the plugins structure should look like::
ase/
    plugins/
            myplugin/
                      __init.py__
                      whatever_you_like.py

The following command just tell to the python, that during import he should
look not only into the ase distribution package, but to the others distribution
packages as well.
Unfortunatelly (it's not documented well in the link above), it's not sufficient
to make the ase.plugins package to be namespace package, we need to do it for
all packages "on the way up" (so for the ase package as well).
"""
__path__ = __import__('pkgutil').extend_path(__path__, __name__)
