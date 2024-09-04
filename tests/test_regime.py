import numpy as np
from regime import RegimeClassifier
from my_vhmm import MyVariationalGaussianHMM
import pytest


# earthquake data from http://earthquake.usgs.gov/
X = np.array([
    13, 14, 8, 10, 16, 26, 32, 27, 18, 32, 36, 24, 22, 23, 22, 18,
    25, 21, 21, 14, 8, 11, 14, 23, 18, 17, 19, 20, 22, 19, 13, 26,
    13, 14, 22, 24, 21, 22, 26, 21, 23, 24, 27, 41, 31, 27, 35, 26,
    28, 36, 39, 21, 17, 22, 17, 19, 15, 34, 10, 15, 22, 18, 15, 20,
    15, 22, 19, 16, 30, 27, 29, 23, 20, 16, 21, 21, 25, 16, 18, 15,
    18, 14, 10, 15, 8, 15, 6, 11, 8, 7, 18, 16, 13, 12, 13, 20,
    15, 16, 12, 18, 15, 16, 13, 15, 16, 11, 11
])


@pytest.fixture
def rc() -> RegimeClassifier:
    return RegimeClassifier(n_components=3)


@pytest.fixture
def rc_populated() -> RegimeClassifier:
    rc = RegimeClassifier(n_components=3)
    rc.models.append(
        MyVariationalGaussianHMM(n_components=rc.n_components)
    )
    return rc

def test_init():
    rc = RegimeClassifier(n_components=3)


def test_properties_with_no_models(rc):
    with pytest.raises(AttributeError):
        rc.model
    
    assert isinstance(rc.init_config, dict)
    assert not rc.has_models
    assert not rc.is_fitted
    assert rc.n_components == 3
    assert rc.transition_threshold == np.inf


def test_properties_with_untrained_model(rc_populated):
    rc_populated.model
    assert isinstance(rc_populated.init_config, dict)

    assert rc_populated.has_models
    assert not rc_populated.is_fitted
    assert rc_populated.n_components == 3
    assert rc_populated.transition_threshold == np.inf



def test_properties_with_trained_model(rc_populated):
    rc_populated.model.fit(X[:, None])
    rc_populated.model.transition_cost = 3
    assert isinstance(rc_populated.init_config, dict)

    assert rc_populated.has_models
    assert rc_populated.is_fitted
    assert rc_populated.n_components == 3
    assert rc_populated.transition_threshold == 1.2 * 3


def test_model_setting_and_getting(rc):
    with pytest.raises(AttributeError):
        rc.model
    with pytest.raises(IndexError):
        rc.models[-1]
    with pytest.raises(IndexError):
        rc[-1]

    rc.model = 'A'
    rc.model = 'B'
    rc.model = 'C'
    rc.model = 'D'
    assert rc.has_models

    assert rc.model == 'D'
    assert rc.models[-1] == 'D'
    assert rc[-1] == 'D'

    assert rc[0] == 'A'
    assert rc[1] == 'B'
    assert rc[2] == 'C'
    assert rc[3] == 'D'
    
    assert rc.models[0] == 'A'
    assert rc.models[1] == 'B'
    assert rc.models[2] == 'C'
    assert rc.models[3] == 'D'


def test_initial_fit():
    rc = RegimeClassifier(n_components=2, n_iter=5)
    rc.initial_fit(X[:, None])
    assert rc.n_components == 2

    rc = RegimeClassifier(n_components=[2,3], n_iter=5)
    rc.initial_fit(X[:, None])

    rc = RegimeClassifier(n_components=-1, n_iter=2)
    rc.initial_fit(X[:, None])


def test_fit():
    rc = RegimeClassifier(n_components=2, n_iter=5)
    rc.initial_fit(X[:50, None])

    rc.fit(X[:, None])
