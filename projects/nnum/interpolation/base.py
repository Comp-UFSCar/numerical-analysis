import abc


class Interpolation(metaclass=abc.ABCMeta):
    def __init__(self, degree):
        """Base Interpolation class.

        :type degree: int
            degree of the interpolation performed must be smaller than number of samples X fitted.
        """
        self.degree = degree

    def fit(self, X):
        self.X = X

    def predict(self, t):
        raise NotImplementedError
