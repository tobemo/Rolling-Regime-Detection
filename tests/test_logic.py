import pytest
from my_vhmm import MyVariationalGaussianHMM
from regime import get_transition_cost_matrix, match_regimes

import numpy as np


np.random.seed(6)
n_samples_per_regime = [300, 100, 100, 100]


@pytest.fixture
def X() -> np.ndarray:
    means = [
        np.array([0.0, 0.0]),    # Mean of regime 1
        np.array([5.0, 5.0]),    # Mean of regime 2
        np.array([10.0, 0.0]),   # Mean of regime 3
        np.array([0.0, 10.0])    # Mean of regime 4
    ]

    covariances = [
        np.array([[1.0, 0.2], [0.2, 1.0]]),   # Covariance of regime 1
        np.array([[1.5, -0.4], [-0.4, 1.0]]), # Covariance of regime 2
        np.array([[0.8, 0.3], [0.3, 1.2]]),   # Covariance of regime 3
        np.array([[1.0, 0.0], [0.0, 1.5]])    # Covariance of regime 4
    ]
    X = np.vstack([
            np.random.multivariate_normal(m, c, n)
            for m, c, n in zip(means, covariances, n_samples_per_regime)
    ])
    return X

@pytest.fixture
def Xs(X) -> list[np.ndarray]:
    return [X[:i] for i in np.cumsum(n_samples_per_regime)[1:]]


@pytest.fixture
def Z() -> np.ndarray:
    Z = np.hstack([
        np.full(n, i)
        for i, n in enumerate(n_samples_per_regime)
    ])
    return Z

@pytest.fixture
def Zs(Z) -> list[np.ndarray]:
    return [Z[:i] for i in np.cumsum(n_samples_per_regime)[1:]]


@pytest.fixture
def model_0(Xs, Zs) -> MyVariationalGaussianHMM:
    model = MyVariationalGaussianHMM(n_components=2)
    model.fit(Xs[0])
    return model


def test_map(model_0, Xs, Zs):
    X, Z = Xs[0], Zs[0]
    for i in range(10):
        model_1 = MyVariationalGaussianHMM(n_components=2, random_state=i)
        model_1.fit(X)
        tcm = get_transition_cost_matrix(
            old_regimes=model_0.predict(X),
            new_regimes=model_1.predict(X),
            n_old_regimes=model_0.n_components,
            n_new_regimes=model_1.n_components,
            data=X
        )
        mapping = match_regimes(tcm)
        if mapping[0][0] != mapping[0][1]:
            break
    
    model_1.mapping = mapping
    assert model_1.mapping[0][0] != model_1.mapping[0][1], "Rest of test doesn't make sense if this is not the case."

    Z_ = model_1.predict(X)
    assert Z_ == pytest.approx(model_0.predict(X))
