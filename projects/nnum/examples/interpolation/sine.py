import matplotlib.pyplot as plt
import numpy as np
from projects.nnum.interpolation.lagrangian import LagrangianInterpolator

interpolation_degree = 2
n_samples = 20
n_predictions = 1000
n_interval = 19


def main():
    print('Interpolating the Sine Function')

    X_train = np.zeros((n_samples, 1))
    for i in range(n_samples):
        X_train[i, 0] = i

    y_train = np.array([np.sin(np.pi / 4 * i) for i in range(n_samples)])
    X_test = np.random.rand(n_predictions, 1) * n_interval

    i = LagrangianInterpolator(interpolation_degree,
                               fitting_profile='local').fit(X_train, y_train)

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
