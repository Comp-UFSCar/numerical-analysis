import abc


class Interpolation(metaclass=abc.ABCMeta):
    def __init__(self, degree):
        """Base Interpolation class.

        :type degree: int
            degree of the interpolation performed must be smaller than number of samples X fitted.
        """
        self.degree = degree
        self.X = self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

        return self

    def predict(self, t):
        raise NotImplementedError
