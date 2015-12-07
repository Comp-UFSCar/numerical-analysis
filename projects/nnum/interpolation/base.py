import abc
from ..base import Predictor


class Interpolator(Predictor, metaclass=abc.ABCMeta):
    """Base Interpolator class.

    Interpolates a function based on the points expressed
    by X and y.

    :type degree, int
        Degree of the interpolation performed must
        be smaller than number of samples X fitted.
    """

    def __init__(self, degree: int):
        super().__init__()

        self.degree = degree
        self.X = self.y = None
