"""

==============================
Interpolating the Sine function.

* Interpolating the sine function using Lagrangian and Newton's methods.
==============================
"""
import matplotlib.pyplot as plt
import numpy as np
from projects.nnum.interpolation.lagrangian import LagrangianInterpolator

print(__doc__)

interpolation_degree = 3
n_samples = 5
n_predictions = 400
n_interval = 4

f = np.log
x = lambda i: i  # np.pi / 4 * i

X_train = np.zeros((n_samples, 1))
for i in range(n_samples):
    X_train[i, 0] = x(i)

y_train = np.array([f(X_train[i]) for i in range(n_samples)])
X_test = x(np.random.rand(n_predictions, 1) * n_interval)

i = LagrangianInterpolator(interpolation_degree,
                           fitting_profile='local').fit(X_train, y_train)

y_pred = np.array([i.predict(t) for t in X_test])

figure = plt.figure(figsize=(16, 9))
plt.scatter(X_test.flatten(), y_pred, color='crimson')
plt.scatter(X_train.flatten(), y_train, color='orange', s=200)
plt.axis('tight')
plt.show()
