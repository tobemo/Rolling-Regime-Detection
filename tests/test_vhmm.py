from my_vhmm import MyVariationalGaussianHMM
import pytest
import numpy as np


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
def trained_model() -> MyVariationalGaussianHMM:
    obj = MyVariationalGaussianHMM(n_components=4, n_iter=10)
    obj.fit(X[:, None])
    return obj


def test_init():
    obj = MyVariationalGaussianHMM()


def test_check(trained_model):
    # check if self._check is disabled for fitted models
    assert trained_model.is_fitted
    # use means prior to test for error
    trained_model.means_prior_ = np.array([])
    trained_model.means_posterior_ = np.array([])
    trained_model._check()

    # check if self._check is not disabled for unfitted models
    obj = MyVariationalGaussianHMM(**trained_model.init_config)
    obj.n_features = trained_model.n_features
    obj.means_prior_ = np.array([])
    obj.means_posterior_ = np.array([])
    with pytest.raises(ValueError):
        obj._check()


def test_fit(trained_model):
    trained_model.predict(X[:, None]) 


def test_loading_config(trained_model):
    cfg = trained_model.get_config()
    obj2 = MyVariationalGaussianHMM.from_config(cfg)
    cfg2 = obj2.get_config()
    cfg.pop('init_params')
    cfg2.pop('init_params')
    assert cfg == cfg2


def test_loading_config_model_equality(trained_model):
    cfg = trained_model.get_config()
    obj2 = MyVariationalGaussianHMM.from_config(cfg)
    assert obj2.means_ == pytest.approx(trained_model.means_)
    assert obj2.covars_ == pytest.approx(trained_model.covars_)
    assert obj2.startprob_ == pytest.approx(trained_model.startprob_)
    assert obj2.transmat_ == pytest.approx(trained_model.transmat_)


def test_loading_config_prediction_equality(trained_model):
    cfg = trained_model.get_config()
    obj2 = MyVariationalGaussianHMM.from_config(cfg)

    y0 = trained_model.predict(X[:, None])
    y1 = obj2.predict(X[:, None])
    assert y1 == pytest.approx(y0)


def test_to_json(trained_model):
    string = trained_model.to_json()


def test_from_json(trained_model):
    string = trained_model.to_json()
    obj2 = MyVariationalGaussianHMM.from_json(string)

    y0 = trained_model.predict(X[:, None])
    y1 = obj2.predict(X[:, None])
    assert y1 == pytest.approx(y0)

    # with mapping set
    trained_model.mapping = np.array([[0, 1, 2, 3], [1, 2, 3, 0]]).T
    string = trained_model.to_json()
    obj2 = MyVariationalGaussianHMM.from_json(string)

    y0 = trained_model.predict(X[:, None])
    y1 = obj2.predict(X[:, None])
    assert y1 == pytest.approx(y0)


def test_transition_cost(trained_model):
    assert trained_model.transition_cost == np.inf
    trained_model.transition_cost = 3.
    trained_model.transition_cost


def test_setting_of_mapping(trained_model):
    # default map is just an arange
    current_map = trained_model.mapping
    map = np.stack(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3]
        ]
    ).T
    assert map == pytest.approx(current_map)

    # wrong shape
    map = np.stack(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3]
        ]
    )
    with pytest.raises(ValueError):
        trained_model.mapping = map
    # right shape
    map = map.T
    trained_model.mapping = map

    # wrong values
    map = np.stack(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4]
        ]
    ).T
    with pytest.raises(ValueError):
        trained_model.mapping = map
    # right values
    map = np.stack(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0]
        ]
    ).T
    trained_model.mapping = map


def test_mapper(trained_model):
    map = np.stack(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0]
        ]
    ).T
    trained_model.mapping = map

    mapper = trained_model.mapper
    assert np.array(list(mapper.keys())) == pytest.approx(map[:,0])
    assert np.array(list(mapper.values())) == pytest.approx(map[:,1])


def test_mapped_predictions():
    model = MyVariationalGaussianHMM(n_components=4)
    model.n_features = 1
    model.startprob_ = np.array([0.4, 0.3, 0.3])
    model.transmat_ = np.array([[0.1, 0.7, 0.2],
                                [0.2, 0.1, 0.7],
                                [0.7, 0.2, 0.1]])
    model.means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
    model.covars_ = 0.5 *  np.tile(np.identity(2), (3, 1, 1))

    X, y0 = model.sample(20)

    y0 = model.predict(X)
    model.mapping = np.stack(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0]
        ]
    ).T
    y1 = model.predict(X)
    assert not y1 == pytest.approx(y0)


def test_equality(trained_model):
    assert trained_model == trained_model
    assert not trained_model == MyVariationalGaussianHMM()
