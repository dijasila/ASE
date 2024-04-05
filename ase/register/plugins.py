""" This module contains the classes for listing plugins and
the plugables (Calculators, IOs, etc...) provided by the plugins.


The structure is as follows


-----------                     --------------------------------
|         |                     |                              |
| Plugins |--------1:n----------|   Plugin (can provide some   |
|         |  (for each plugin   |   calculator, formats...)    |
|         |      package)       |                              |
^^^^^^^^^^                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                                           |
   1:n  for each type                         1:n - plugin can return lists
    |   (i.e viewer, calculator)                | of plugables
    |                                           | of each plugable type
    |                                           | (calculator, viewer)
    |                                           | that it provides
    |                                           |
    |                                           |
    |                                           |
-------------------------                ------------------------
|                       |                |                      |
|    Plugables:         |                |    Plugable:         |
|      list all the     |------1:n-------|    holds inform.     |
|   available 'items'   |                |    about just one    |
|   (calcs, viewers)    |                |    calculator        |
|   of the given type   |                |    or viewer         |
|                       |                |                      |
^^^^^^^^^^^^^^^^^^^^^^^^^                ^^^^^^^^^^^^^^^^^^^^^^^^
                                                  |
                                                 1:1
                                                  |
                                         ------------------------
                                         |                      |
                                         |   Implementation     |
                                         |   (e.g. a Calculator |
                                         |    subclass)         |
                                         |                      |
                                         """"""""""""""""""""""""
"""

from contextlib import contextmanager
import importlib
import pkgutil
import warnings
from ase.utils import lazyproperty
from .listing import Listing
_current_plugin = None


@contextmanager
def within_the_plugin(plugin):
    """ When a plugin register() function is called,
    the registered plugin would have to say which plugin
    am I.
    This can lead to errors, so importing of the plugin
    is enclosed by this helper, due to the
    :func:get_currently_registered_plugin
    can say, which plugin is currently imported.
    """

    global _current_plugin
    ocp = _current_plugin
    _current_plugin = plugin
    yield
    _current_plugin = ocp


def get_currently_registered_plugin():
    """
    Which plugin is imported and so to which plugin
    belongs currently imported Plugables

    See the :func:within_the_plugin
    """
    plugin = _current_plugin
    if not plugin:
        import ase.plugins as ase_plugins
        plugin = ase_plugins.plugins['external']
    return plugin


class Plugins(Listing):
    """ A class, that holds all the installed plugins in
        the given namespace package."""

    """ This information is just for initial creating of the plugables """
    def __init__(self, namespace_package, plugable_types):
        self.namespace_package = namespace_package
        self._plugables = {
            k: cls(self, k)
            for k, cls in plugable_types.items()
        }

    def packages(self):
        """ Return all the plugin packages, that are in the
        given namespace package (so that are in 'ase.plugins') """
        modules = []
        package = importlib.import_module(self.namespace_package)

        def mod_name(mod):
            return self.namespace_package + '.' + mod.name

        def import_plugin_module(name, path):
            try:
                module = importlib.import_module(name)
                return module
            except ImportError:
                warnings.warn(f"Can not import {name} in {path}."
                              " This ASE plugin is probably broken.")

        modules = (import_plugin_module(mod_name(mod), mod.module_finder.path)
                   for mod in pkgutil.iter_modules(package.__path__))

        modules = (i for i in modules if i)
        return modules

    def populate(self):
        self._items = {p.__name__.rsplit('.', 1)[-1]: Plugin(self, p)
                       for p in self.packages()}

    def plugables_of(self, class_type):
        return self._plugables[class_type]

    def all_plugables(self):
        return self._plugables.values()

    @lazyproperty
    def calculators(self):
        return self.plugables_of('calculators')

    @lazyproperty
    def viewers(self):
        return self.plugables_of('viewers')

    @lazyproperty
    def io_formats(self):
        return self.plugables_of('io_formats')

    def __repr__(self):
        return f"<ASE plugins from: {self.namespace_package}>"

    def info(self, prefix='', opts={}):
        return "Plugins:\n"\
               "--------\n" + super().info(prefix, opts)

    def register(self):
        """
        Register all the installed plugables. To do so
        - import all plugin packages
        - register all the plugables from the plugins
        """
        self.populate()
        for i in self:
            i.register()


class Plugin:
    """ A class, that encapsulates a plugin package """

    def __init__(self, plugins, package):
        self.plugins = plugins
        self.package = package
        self.plugables = {
            i.class_type: {} for i in plugins.all_plugables()
        }
        self.modules = {}
        self._plugables = {}
        self.registered = False

    def add_plugable(self, plugable):
        """ Called by Plugable.register() """
        self.plugables[plugable.class_type][plugable.name] = plugable

    @property
    def name(self):
        return self.package.__name__[12:]   # get rig 'ase.plugins'

    def register(self):
        """ Register the plugables in the plugin:
        Call the ase_register function from the
        __init__.py of the plugin package
        """
        if not self.registered:
            if hasattr(self.package, 'ase_register'):
                with within_the_plugin(self):
                    self.package.ase_register()
            self.registered = True

    def info(self, prefix='', opts={}):
        info = f'{prefix}{self.name}'

        prefix += '  '
        opts = opts.copy()
        opts['plugin'] = False

        for plugables in self.plugins.all_plugables():
            itype = plugables.class_type
            inst = self.plugables[itype]
            if inst:
                p = f'\n{prefix}{plugables.singular_name}: '
                for i in inst.values():
                    info += i.info(p, opts)
        return info

    def __repr__(self):
        return f"<ASE plugin: {self.package}>"
