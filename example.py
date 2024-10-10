import logging

import numpy as np
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

from matplotlib import pyplot as plt

from .hmm.vghmm import VariationalGaussianHMM
from .plotting_utils import plot_multiple_with_regimes


logging.basicConfig(level=logging.DEBUG)
from .regime import RegimeClassifier


np.random.seed(4)

regimes = [0, 1, 2, 1, 3]
n_samples_per_regime = [300, 100, 99, 100, 200]
colors = {i: plt.cm.Set3(i / len(regimes)) for i in regimes}
colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'purple'}


means = [0, 5, 10, 5, 20]
covariances = [
    np.array([1.0]),   # Covariance of regime 1
    np.array([0.2]), # Covariance of regime 2
    np.array([1.8]),   # Covariance of regime 3
    np.array([0.2]), # Covariance of regime 2
    np.array([1.0]),    # Covariance of regime 4
]

X = np.concatenate([
        np.random.normal(m, size=n)#.clip(m - 5 * c, m + 5 * c)
        for m, c, n in zip(means, covariances, n_samples_per_regime)
])
X[281] = X[280] # fix for seed=4
Z = np.hstack([
    np.full(n, i)
    for i, n in zip(regimes, n_samples_per_regime)
])

Xs = [X[:i] for i in np.cumsum(n_samples_per_regime)[1:]]
Zs = [Z[:i] for i in np.cumsum(n_samples_per_regime)[1:]]


rc = RegimeClassifier(n_components=[2,3,4])
rc.fit(Xs[0][:, None])
rc.fit(Xs[1][:, None])
rc.fit(Xs[2][:, None])
rc.fit(Xs[3][:, None])


Zs_ = [model.predict(X[:, None]) for model, X in zip(rc.models, Xs)]
plot_multiple_with_regimes(Xs, Zs_, regime_colors=colors)
plt.show()