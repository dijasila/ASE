from ase.register.plugins import get_currently_registered_plugin

# import them later to avoid a circular import
define_io_format = None
define_viewer = None


def _register(plugable_type: str, cls: str, name=None):
    """ Register a calculator or other plugable exposed by a plugin.
    The name can be derived from the cls name

    Parameters
    ----------
    plugable_type
      Which to register. E.g. 'calculators', 'io_formats' and so on.
      However, io_formats have its old routine to register, so they
      do not use this mechanism.

    cls: str
      Which class implements the plugable (e.g. 'ase.plugins.emt.EMTCalculator')
      The class goes by its name only to avoid importing too myuch stuff.
    """
    if not name:
        name = cls.rsplit(".", 1)[-1]
    plugin = get_currently_registered_plugin()
    p_cls = plugin.plugins.plugables_of(plugable_type).item_type
    plugable = p_cls(plugin, plugable_type, name, cls)
    plugable.register()


def register_calculator(cls: str, name=None):
    """ Register a calculator exposed by a plugin.
    The name can be derived from the cls name
    """
    _register('calculators', cls, name)


def register_io_format(module, desc, code, *, name=None, ext=None,
                       glob=None, magic=None, encoding=None,
                       magic_regex=None):
    """ Just a wrapper for :func:`ioformats.define_io_format`.
    The order of parameters is however slightly different here,
    to be as much as possible similiar to the :func:`register_calculator`
    """
    global define_io_format
    if not define_io_format:
        from ase.io.formats import define_io_format
    if not name:
        name = module.rsplit(".", 1)[-1]
    fmt = define_io_format(name, desc, code,
                           module=module,
                           ext=ext,
                           glob=glob,
                           magic=magic,
                           encoding=encoding,
                           magic_regex=magic_regex,
                           external=True)
    fmt.plugin = get_currently_registered_plugin()
    return fmt


def register_viewer(name, desc, *, module=None, cli=False, fmt=None, argv=None):
    if not define_viewer:
        from ase.visualize.viewers import define_viewer
    view = define_viewer(name, desc, module=module, cli=cli,
                         fmt=fmt, argv=argv, external=True)
    view.plugin = get_currently_registered_plugin()
    return view


__all__ = ['register_viewer', 'register_io_format', 'register_calculator']
