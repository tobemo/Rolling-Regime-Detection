import numpy as np
import pandas as pd
import pytest

from rolling.hmm.vghmm import VariationalGaussianHMM
from rolling.utils import (added_regime_costs_less, calculate_total_cost,
                           copy_model, extend_startprob, extend_transmat,
                           get_regime_map, get_transition_cost_matrix,
                           new_regime_is_advised, old_regime_is_too_costly,
                           sample_by)


def test_extend_transmat():
    transmat = np.array([[0.3, 0.4, 0.3, 0.0],
                         [0.1, 0.2, 0.7, 0.0],
                         [0.5, 0.2, 0.3, 0.0],
                         [0.2, 0.2, 0.6, 0.0]])
    # last col should be min of each row
    # last row should be all equal
    # each row should sum to 1
    expected = np.array([[0.3, 0.4, 0.3, 0.0, 0.0],
                         [0.1, 0.2, 0.7, 0.0, 0.0],
                         [0.5, 0.2, 0.3, 0.0, 0.0],
                         [0.2, 0.2, 0.6, 0.0, 0.0],
                         [0.2, 0.2, 0.2, 0.2, 0.2]])
    extended = extend_transmat(transmat, 1)
    assert extended == pytest.approx(expected)



def test_extend_startprob():
    startprob = np.array([0.1, 0.4, 0.3, 0.2])
    # new regime has startprob equal to smallest one
    # already present
    expected =  np.array([0.1, 0.4, 0.3, 0.2, 0.1])
    expected /= expected.sum()
    extended = extend_startprob(startprob, 1)
    assert extended == pytest.approx(expected)


def test_transition_cost_matrix():
    size = 10
    n_regimes = 2
    data = np.random.rand(size)
    old_regimes = np.random.randint(low=0, high=n_regimes, size=size)
    new_regimes = old_regimes.copy()

    # old regimes < new regimes
    with pytest.raises(ValueError):
        get_transition_cost_matrix(
            old_regimes=old_regimes,
            new_regimes=new_regimes,
            n_old_regimes=n_regimes+1,
            n_new_regimes=n_regimes,
            data=data,
        )
    # old regimes == new regimes
    get_transition_cost_matrix(
        old_regimes=old_regimes,
        new_regimes=new_regimes,
        n_old_regimes=n_regimes,
        n_new_regimes=n_regimes,
        data=data,
    )
    # old regimes > new regimes
    get_transition_cost_matrix(
        old_regimes=old_regimes,
        new_regimes=new_regimes,
        n_old_regimes=n_regimes,
        n_new_regimes=n_regimes+5,
        data=data,
    )

    # correctness of the diagonal
    # when regimes are equal
    tcm = get_transition_cost_matrix(old_regimes, new_regimes, n_regimes, n_regimes, data)
    assert np.diag(tcm) == pytest.approx(np.zeros(n_regimes))

    data = np.ones(size)
    old_regimes = np.random.randint(low=0, high=2, size=size)
    new_regimes = old_regimes.copy()
    data[old_regimes == 1] = 2
    tcm = get_transition_cost_matrix(old_regimes, new_regimes, n_regimes, n_regimes, data)
    assert np.diagonal(np.flipud(tcm)) == pytest.approx(np.ones(n_regimes))


def test_match_regimes():
    a = np.array([[0., 1],
                 [1, 0.]])
    expected = np.array([[0, 0],
                         [1, 1]])
    match = get_regime_map(a)
    assert match == pytest.approx(expected)

    b = np.array([[0.5, 0.0, 0.5],
                  [0.0, 0.5, 0.5],
                  [0.5, 0.5, 0.5]])
    expected = np.array([[0, 1],
                         [1, 0],
                         [2, 2]])
    match = get_regime_map(b)
    assert match == pytest.approx(expected)


def test_calculate_total_cost():
    tcm = np.array([[0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.5, 0.0]])
    cost = calculate_total_cost(tcm)
    assert cost == 0

    tcm = np.array([[0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.5, 0.5]])
    cost = calculate_total_cost(tcm, norm=3)
    assert cost == 0.5 / 3
    
    tcm = np.array([[0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [np.inf, np.inf, np.inf]])
    cost = calculate_total_cost(tcm, norm=3)
    assert cost == 0.5 / 3


def test_new_regime_is_advised():
    assert added_regime_costs_less(
        regime_cost=2,
        added_regime_cost=1
    )

    assert old_regime_is_too_costly(
        regime_cost=2,
        threshold=1
    )

    assert new_regime_is_advised(
        regime_cost=3,
        added_regime_cost=2,
        threshold=2.5,
    )


def test_sampling_by():
    # mismatch X & Z
    with pytest.raises(ValueError):
        sample_by(
            X=np.arange(3),
            Z=np.arange(10),
            f=0.5
        )
    with pytest.raises(TypeError):
        sample_by(
            X=np.arange(10),
            Z=pd.Series(np.arange(10)),
            f=0.5
        )
    with pytest.raises(ValueError):
        sample_by(
            X=pd.Series(np.arange(10)),
            Z=pd.Series(np.arange(10), index=pd.RangeIndex(1, 12)),
            f=0.5
        )
    sample_by(
        X=np.arange(10),
        Z=np.arange(10),
        f=0.5
    )

    # mismatch sample probability
    with pytest.raises(ValueError):
        sample_by(
            X=np.arange(10),
            Z=np.arange(10),
            f=0.5,
            p=np.arange(3)
        )
    with pytest.raises(ValueError):
        sample_by(
            X=pd.Series(np.arange(10)),
            Z=pd.Series(np.arange(10)),
            f=0.5,
            p=pd.Series(np.arange(10), index=pd.RangeIndex(1, 12)),
        )
    
    X = np.array([0]*20 + [1,1])
    Z = np.array([0]*20 + [1, 1])
    # check per regime sampling
    s = sample_by(
        X=X,
        Z=Z,
        f=0.1,
        p=np.linspace(0,1, 22),
    )
    assert s[-1] == 1

    # with per sample proba where proba for one or more groups is 0
    s = sample_by(
        X=X,
        Z=Z,
        f=0.1,
        p=np.concatenate([np.linspace(0,1, 20), np.zeros(2)])
    )
    assert s[-1] == 0


def test_copy():
    X = np.array([
        13, 14, 8, 10, 16, 26, 32, 27, 18, 32, 36, 24, 22, 23, 22, 18,
        25, 21, 21, 14, 8, 11, 14, 23, 18, 17, 19, 20, 22, 19, 13, 26,
        13, 14, 22, 24, 21, 22, 26, 21, 23, 24, 27, 41, 31, 27, 35, 26,
        28, 36, 39, 21, 17, 22, 17, 19, 15, 34, 10, 15, 22, 18, 15, 20,
        15, 22, 19, 16, 30, 27, 29, 23, 20, 16, 21, 21, 25, 16, 18, 15,
        18, 14, 10, 15, 8, 15, 6, 11, 8, 7, 18, 16, 13, 12, 13, 20,
        15, 16, 12, 18, 15, 16, 13, 15, 16, 11, 11
    ])
    a = VariationalGaussianHMM(name='foo')
    a.fit(X[:, None])
    b = copy_model(a)
    assert a == b

    b.name = 'bar'
    assert a != b

