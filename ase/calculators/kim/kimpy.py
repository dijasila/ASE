from ase.utils import lazyproperty
from typing import Dict


class KimpyLazyFunction:
    """ Function, that wraps a kimpy function, allows the import of kimpy module
    only when it is really called """

    def __init__(self, name):
        self.name = name

    @lazyproperty
    def fn(self):
        fn = kimpy
        for i in self.name.split('.'):
            fn = getattr(fn, i)
        return fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class KimpyLazyFunctions:
    """ Lazy functions holds all the lazy functions, to be sure to be
    created just once """
    def __getattr__(self, name):
        fn = KimpyLazyFunction(name)
        setattr(self, name, fn)
        return fn


class Kimpy:
    """ Just a wrapper, that wraps kimpy to allow a lazy import """

    # just a hack to make lazyproperty do not interfere with __getattr__
    _lazy_cache: Dict = {}

    @property
    def kimpy(self):
        try:
            import kimpy
        except ImportError:
            raise NotImplementedError("To use KIM calculator, please "
                                      "install kimpy package")
        self.__dict__['kimpy'] = kimpy
        return kimpy

    def __getattr__(self, name):
        return getattr(self.kimpy, name)

    @lazyproperty
    def lazy_functions(self):
        return KimpyLazyFunctions()


kimpy = Kimpy()
