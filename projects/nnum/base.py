import abc


class Fitter:
    def __init__(self):
        self.X, self.y = None, None

    def fit(self, X, y):
        self.X = X
        self.y = y

        return self


class Predictor(Fitter, metaclass=abc.ABCMeta):
    def predict(self, X):
        raise NotImplementedError
