import matplotlib.pyplot as plt
import numpy as np
from projects.nnum.interpolation.lagrangian import LagrangianInterpolation

interpolation_degree = 20
n_samples = 20
n_predictions = 100
n_interval = 19


def main():
    print('Random Interpolation Example')

    X_train = np.zeros((n_samples, 1))
    for i in range(n_samples):
        X_train[i, 0] = i

    y_train = np.array([i ** 2 for i in range(n_samples)])
    # y_train = np.random.rand(n_samples) * 100
    X_test = np.random.rand(n_predictions, 1) * n_interval

    i = LagrangianInterpolation(interpolation_degree,
                                fitting_profile='sparse').fit(X_train, y_train)

    y_pred = np.array([i.predict(t) for t in X_test])
    print(y_pred)

    figure = plt.figure(figsize=(16, 9))
    plt.scatter(X_train.flatten(), y_train, color='red')
    plt.scatter(X_test.flatten(), y_pred, color='orange')
    plt.axis('tight')
    plt.show()

    print('Done.')


if __name__ == '__main__':
    main()
