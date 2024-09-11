import pytest
from my_vhmm import MyVariationalGaussianHMM
from regime import get_transition_cost_matrix, get_regime_map, calculate_total_cost

import numpy as np


np.random.seed(6)
n_samples_per_regime = [300, 100, 100, 100]


@pytest.fixture
def X() -> np.ndarray:
    means = [0, 5, 10, 5, 30]

    covariances = [
        np.array([1.0]),   # Covariance of regime 1
        np.array([0.2]),   # Covariance of regime 2
        np.array([0.8]),   # Covariance of regime 3
        np.array([0.2]),   # Covariance of regime 2
        np.array([1.0]),   # Covariance of regime 4
    ]
    Xs = np.concatenate([
        np.random.normal(m, size=n)
        for m, c, n in zip(means, covariances, n_samples_per_regime)
    ])
    return Xs[:, None]

@pytest.fixture
def Xs(X) -> list[np.ndarray]:
    """
    X[0]: 2 regimes;
    X[1]: 3 regimes;
    X[2]: 4 regimes;
    """
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
    """A model with 2 components."""
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
        mapping = get_regime_map(tcm)
        if mapping[0][0] != mapping[0][1]:
            break
    
    model_1.mapping = mapping
    assert model_1.mapping[0][0] != model_1.mapping[0][1], "Rest of test doesn't make sense if this is not the case."

    Z_ = model_1.predict(X)
    assert Z_ == pytest.approx(model_0.predict(X))


def _test_mapping_long_term(model_0, Xs):
    # ! ABANDONED, writing tests for unsupervised models is hard
    model_a = model_0
    for X in Xs[1:]:
        # add a regime each iteration
        model_b = MyVariationalGaussianHMM(
            n_components=model_a.n_components + 1
        )
        model_b.fit(X)

        tcm = get_transition_cost_matrix(
            old_regimes=model_a.predict(X),
            new_regimes=model_b.predict(X),
            n_old_regimes=model_a.n_components,
            n_new_regimes=model_b.n_components,
            data=X
        )

        model_b.mapping = get_regime_map(tcm)
        model_a = model_b
    
    # TODO: I don't think mapping holds true over time
    # check mapping over time
    X = Xs[0]
    tcm = get_transition_cost_matrix(
        old_regimes=model_0.predict(X),
        new_regimes=model_a.predict(X),
        n_old_regimes=model_0.n_components,
        n_new_regimes=model_a.n_components,
        data=X
    )
    mapping = get_regime_map(tcm)
    pass


def train_model(n_components: int, X: np.ndarray) -> MyVariationalGaussianHMM:
    _score = -np.inf
    model = None
    for i in range(5):
        _model = MyVariationalGaussianHMM(
            n_components=n_components,
            random_state=i
        )
        _model.fit(X)
        if _model.score(X) > _score:
            _score = _model.score(X)
            model = _model
    model = model or _model
    return model   


def _test_increasing_regime(model_0, Xs):
    # ! ABANDONED, writing tests for unsupervised models is hard
    X_3_regimes = Xs[1]
    model_2_regimes = train_model(
        n_components=model_0.n_components,
        X=X_3_regimes,
    )
    model_3_regimes = train_model(
        n_components=model_0.n_components + 1,
        X=X_3_regimes,
    )

    tcm_2_regimes = get_transition_cost_matrix(
        old_regimes=model_0.predict(X_3_regimes),
        new_regimes=model_2_regimes.predict(X_3_regimes),
        n_old_regimes=model_0.n_components,
        n_new_regimes=model_2_regimes.n_components,
        data=X_3_regimes,
    )
    cost_2_regimes = calculate_total_cost(tcm_2_regimes)
    tcm_3_regimes = get_transition_cost_matrix(
        old_regimes=model_0.predict(X_3_regimes),
        new_regimes=model_3_regimes.predict(X_3_regimes),
        n_old_regimes=model_0.n_components,
        n_new_regimes=model_3_regimes.n_components,
        data=X_3_regimes,
    )
    cost_3_regimes = calculate_total_cost(tcm_3_regimes)
    assert cost_3_regimes < cost_2_regimes

    # iterate all x
    # check if last model has similar predict on X[0] as model_0
        # only for first 2 regimes
        # comparable mean and std
        # or low distance
