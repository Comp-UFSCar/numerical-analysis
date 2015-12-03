import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tests.test_style import temp_style
from mpl_toolkits.mplot3d import Axes3D

Axes3D

from projects.nnum.interpolation.lagrangian import LagrangianInterpolation

interpolation_degree = 3
n_samples = 10
n_predictions = 100


def main():
    print('Random Interpolation Example')

    X = np.zeros((n_samples, 1))
    for i in range(n_samples):
        X[i, 0] = i

    y = np.random.rand(n_samples) * 100

    i = LagrangianInterpolation(interpolation_degree).fit(X, y)
    tst_data = np.random.rand(n_predictions, 1)
    y_predicted = np.array([i.predict(t) for t in tst_data])

    print(y_predicted)

    figure = plt.figure(figsize=(16, 9))

    plt.scatter(X.flatten(), y, color='red')
    plt.scatter(tst_data.flatten(), y_predicted, color='orange')
    plt.axis('tight')

    plt.show()


if __name__ == '__main__':
    main()
