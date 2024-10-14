import json

import numpy as np
import pandas as pd
import pytest

from hmm.vghmm import VariationalGaussianHMM
from regime import RegimeClassifier


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
ts = pd.date_range(end=pd.Timestamp.now(), freq='D', periods=len(X))


@pytest.fixture
def rc() -> RegimeClassifier:
    return RegimeClassifier(n_components=3)


@pytest.fixture
def rc_populated() -> RegimeClassifier:
    rc = RegimeClassifier(n_components=3)
    rc.models.append(
        VariationalGaussianHMM(n_components=rc.n_components)
    )
    return rc


@pytest.fixture
def rc_fitted() -> RegimeClassifier:
    rc = RegimeClassifier(n_components=2)
    rc.fit(X[:20, None])
    rc.fit(X[:, None])
    return rc


def test_init():
    rc = RegimeClassifier(n_components=3)
    str(rc)
    repr(rc)


def test_properties_with_no_models(rc):
    with pytest.raises(AttributeError):
        rc.model_
    
    assert isinstance(rc.get_params(), dict)
    assert not rc.has_models
    assert not rc.is_fitted
    assert rc.n_components == 3
    assert rc.transition_threshold == np.inf


def test_properties_with_untrained_model(rc_populated):
    rc_populated.model_
    assert isinstance(rc_populated.get_params(), dict)

    assert rc_populated.has_models
    assert not rc_populated.is_fitted
    assert rc_populated.n_components == 3
    assert rc_populated.transition_threshold == np.inf


def test_properties_with_trained_model(rc_populated):
    rc_populated.model_.fit(X[:, None])
    rc_populated.model_.transition_cost = 3
    assert isinstance(rc_populated.get_params(), dict)

    assert rc_populated.has_models
    assert rc_populated.is_fitted
    assert rc_populated.n_components == 3
    assert rc_populated.transition_threshold == 1.2 * 3


def test_model_setting_and_getting(rc):
    with pytest.raises(AttributeError):
        rc.model_
    with pytest.raises(IndexError):
        rc.models[-1]

    rc.model_ = 'A'
    rc.model_ = 'B'
    rc.model_ = 'C'
    rc.model_ = 'D'
    assert rc.has_models

    assert rc.model_ == 'D'
    assert rc.models[-1] == 'D'

    assert rc.models[0] == 'A'
    assert rc.models[1] == 'B'
    assert rc.models[2] == 'C'
    assert rc.models[3] == 'D'


def test_getitem(rc):
    rc.model_ = 'A'
    rc.model_ = 'B'
    rc.model_ = 'C'
    rc.model_ = 'D'

    rc2 = rc[2]
    assert rc2.model_ == 'C'
    rc2 = rc[-2]
    assert rc2.model_ == 'C'
    rc2 = rc[0]
    assert rc2.model_ == 'A'
    rc2 = rc[-1]
    assert rc2.model_ == 'D'


def test_initial_fit():
    rc = RegimeClassifier(n_components=2, n_iter=5)
    rc._initial_fit(X[:, None])
    assert rc.n_components == 2

    rc = RegimeClassifier(n_components=[2,3], n_iter=5)
    rc._initial_fit(X[:, None])

    rc = RegimeClassifier(n_components=-1, n_iter=2)
    rc._initial_fit(X[:, None])


def test_transition_threshold(rc, rc_populated):
    # no models
    assert rc.transition_threshold == np.inf

    # < 8 models
    rc_populated.model_.transition_cost = 3
    assert rc_populated.transition_threshold == 1.2 * 3
    
    # setting
    rc.transition_threshold = 3
    assert rc.transition_threshold == 3
    rc.transition_threshold = None
    assert rc.transition_threshold == np.inf

    # > 8 models
    model = rc_populated.model_
    model.transition_cost = 3
    for _ in range(8):
        rc.models.append(model)
    assert rc.transition_threshold == 3


def test_fit():
    # np
    rc = RegimeClassifier(n_components=2, n_iter=5)
    rc.fit(X[:50, None])
    rc.fit(X[:, None])
    # pd
    rc = RegimeClassifier(n_components=2, n_iter=5)
    X_ = pd.DataFrame(X)
    rc.fit(X_)
    # pd with datetime index
    rc = RegimeClassifier(n_components=2, n_iter=5)
    X_ = pd.DataFrame(X, index=ts)
    rc.fit(X_)


def test_predict(rc_fitted):
    # np
    rc_fitted._predict(X[:, None])
    # pd
    X_ = pd.DataFrame(X)
    rc_fitted._predict(X_)
    # pd with datetime index
    X_ = pd.DataFrame(X, index=ts)
    rc_fitted._predict(X_)


def test_predict_all(rc_fitted):
    X_ = pd.DataFrame(X)
    rc_fitted.predict_all(X_)
    
    X_ = pd.DataFrame(
        X,
        index=pd.date_range(
            end=pd.Timestamp.now(), freq='D', periods=len(X)
        ),
    )
    
    # test index
    rc_fitted.models[0].timestamp_ = X_.index[2].isoformat()
    rc_fitted.models[1].timestamp_ = X_.index[-3].isoformat()
    y = rc_fitted.predict_all(X_)
    pd.testing.assert_index_equal(y.index, X_.index)
    
    # test nans (negative space)
    y_ = (y*0.).to_numpy() # discard regime info
    expected = np.zeros((len(X), 2)) # 2 models
    expected[2+1:, 0] = None
    expected[-(3-1):, 1] = None
    np.testing.assert_equal(y_, expected)
    
    # test regime values
    y0 = rc_fitted.models[0].predict(X_).to_numpy()
    assert y0[:3] == pytest.approx(y.to_numpy()[:3,0])
    y1 = rc_fitted.models[1].predict(X_).to_numpy()
    assert y1[:-3] == pytest.approx(y.to_numpy()[:-3,1])


def test_equality(rc_fitted, rc):
    assert rc_fitted == rc_fitted
    assert not rc_fitted == rc


def test_json(rc, rc_fitted):
    string = rc_fitted.classifier_to_json()
    assert rc_fitted.get_params() == json.loads(string)

    models = rc_fitted.models_to_jsons()
    assert isinstance(models, list)

    # unpopulated
    classifier_config, configs = rc.to_json()
    rc2 = RegimeClassifier.from_jsons(classifier_config, configs)
    assert rc2 == rc

    # populated
    classifier_config, configs = rc_fitted.to_json()
    rc2 = RegimeClassifier.from_jsons(classifier_config, configs)
    assert rc2 == rc_fitted

    # when n_components was init as a list
    # with no previous models
    rc = RegimeClassifier(n_components=[2,3])
    classifier_config, configs = rc.to_json()
    rc2 = RegimeClassifier.from_jsons(classifier_config, configs)
    assert rc2 == rc
    
    # when n_components was init as a list
    # with previous models
    rc = RegimeClassifier(n_components=[2,3])
    for _ in range(3):
        rc.fit(X[:, None])
    classifier_config, configs = rc.to_json()
    rc2 = RegimeClassifier.from_jsons(classifier_config, configs)
    assert rc2 == rc
