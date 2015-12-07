import abc
from ..base import Fitter


class Integrator(metaclass=abc.ABCMeta):
    """Integrator class.

    Integrates a function expressed by data set X and its value y.
    """

    def __init__(self, f, a=0, b=1, n=10):
        assert f is not None
        self.f = f

        assert b > a
        self.a = a
        self.b = b

        assert n > 0
        self.n = n
        self.h = (b - a) / n

    def x(self, i):
        return self.a + i * self.h

    def integrate(self):
        raise NotImplementedError
