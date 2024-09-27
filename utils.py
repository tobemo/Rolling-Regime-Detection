import numpy as np
import pandas as pd
from scipy import optimize, stats

from base import MyHMM


def extend_startprob(startprob: np.ndarray, extension: int) -> np.ndarray:
    """Extend start probability matrix.
    New probabilities are set to the smallest tvalue already present.

    Args:
        startprob (np.ndarray): Old start probabilities.
        extension (int): How many probabilities to add.

    Returns:
        np.ndarray: Extended start probability vector.
    """
    old_shape = startprob.shape[0]
    new_startprob = np.zeros(old_shape + extension)
    new_startprob[:old_shape] = startprob
    # the new start probability is set to the smallest one already present
    new_startprob[old_shape:] = startprob.min(keepdims=True)
    # normalize to sum to 1
    new_startprob /= new_startprob.sum()
    return new_startprob


def extend_transmat(transmat: np.ndarray, extension: int) -> np.ndarray:
    """Extend transition probability matrix.
    New probabilities are set to the smallest value already present.

    Args:
        transmat (np.ndarray): Old transition probabilities.
        extension (int): How many probabilities to add.

    Returns:
        np.ndarray: Extended transition probability matrix.
    """
    ## transition matrix:
    # rows are the current state (S)
    # columns are the next state
    # each row shows the probability to go from that state 
    # to any of the other states, including itself (the diagonal)
    # [0.7, 0.2, 0.1]   # transitions from state 0
    # [0.3, 0.5, 0.2]   # transitions from state 1
    # [0.4, 0.1, 0.5]   # transitions from state 2
    # e.g. 70% to go from S0 to S0, 20% for S0 to S1, 10% for S0 to S3

    ## extending transition matrix:
    # going from new regime(s) to any existing regime has an equal probability
    # [0.7, 0.2, 0.1]
    # [0.3, 0.5, 0.2]
    # [0.4, 0.1, 0.5]
    # [1.0, 1.0, 1.0]   < new regime
    # going from existing regimes A to new regime(s) B^ is set to the smallest 
    # transition probability from A to any of the old regimes B
    # 
    #                  v new regime, set to the min of each row
    # [0.7, 0.2, 0.1, 0.1]
    # [0.3, 0.5, 0.2, 0.2]
    # [0.4, 0.1, 0.5, 0.1]
    # combined:
    # [0.7, 0.2, 0.1, 0.1]
    # [0.3, 0.5, 0.2, 0.2]
    # [0.4, 0.1, 0.5, 0.1]
    # [1.0, 1.0, 1.0, 1.0]
    # note that afterwards each row is normalized to sum to 1

    old_shape = transmat.shape[0]
    new_transmat = np.zeros(
        (old_shape + extension, old_shape + extension)
    )
    new_transmat[:old_shape, :old_shape] = transmat
    # going from new regime(s) to any existing regime has an equal probability
    new_transmat[old_shape:, :] = 1
    # going from existing regimes A to new regime(s) B^ is set to the 
    # smallest transition probability from A to any of the old regimes B
    new_transmat[:old_shape, old_shape:] = transmat.min(axis=1, keepdims=True)
    # normalize to sum to 1
    new_transmat /= new_transmat.sum(axis=1, keepdims=True)
    return new_transmat


def copy_model(model: MyHMM) -> MyHMM:
    """Copy model as is."""
    config = model.get_fitted_params()
    new_model = type(model).set_fitted_params(config)
    return new_model


def transfer_model(
        old_model: MyHMM,
        n_components: int = None,
        n_iter: int = None,
        tol: float = None,
        ) -> MyHMM:
    """Return a new model of the same type as old model with its transition matrix and start probabilities copied over.
    If `n_component` is greater than `old_model.n_component` then one or more regimes are added."""
    config = old_model.init_config
    config['n_components'] = n_components or config['n_components']
    config['n_iter'] = n_iter or config['n_iter']
    config['tol'] = tol or config['tol']
    
    n_component_difference = config['n_components'] - old_model.n_components
    if n_component_difference < 0:
        raise NotImplementedError("A reduction in the number of regimes is not supported.")
    
    config['init_params'] = 'mc'
    new_model = type(old_model)(**config)

    startprob_ = old_model.startprob_
    transmat_ = old_model.transmat_

    if n_component_difference > 0:
        startprob_ = extend_startprob(
            startprob=startprob_,
            extension=n_component_difference,
        )
        transmat_ = extend_transmat(
            transmat=transmat_,
            extension=n_component_difference,
        )
    
    new_model.startprob_ = startprob_
    new_model.transmat_ = transmat_
    new_model.random_state = None
    return new_model


def get_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Get Wassertsein distance between two distributions datasets."""
    if u.ndim == 2:
        return np.mean(
            [stats.wasserstein_distance(u[i], v[i]) for i in range(u.shape[1])]
        )
    return stats.wasserstein_distance(u, v)


def get_transition_cost_matrix(
        old_regimes: np.ndarray | pd.Series,
        new_regimes: np.ndarray | pd.Series,
        n_old_regimes: int,
        n_new_regimes: int,
        data: np.ndarray | pd.Series,
    ) -> np.ndarray:
    """Returns cost matrix.
    The cost matrix describes the distance between each old and new distribution. The cost is defined as the distance between the distributions of the data first index by an old regime and then index by a new regime.
    E.g. `data[old regime 0] <> data[new regime 0], data[old regime 0] <> data[new regime 1], ..` The cost of each old-new pair is computed and returned.

    Args:
        old_regimes (np.ndarray): Array of int denoting old regimes.
        new_regimes (np.ndarray): Array of int denoting new regimes.
        n_old_regimes (int): The number of old regimes. Can be higher than the highest value of `old_regimes`.
        n_new_regimes (int): The number of new regimes. Can be higher than the highest value of `new_regimes`.
        data (np.ndarray): Data from which regimes are derived.

    Raises:
        ValueError: `n_new_regimes` shouldn't be lower then `n_old_regimes`.

    Returns:
        np.ndarray: Cost matrix. Rows denote the old regimes, columns the new. A row-column pair then denotes the cost of labeling the new regime the same value as the old one.
    """
    if n_new_regimes < n_old_regimes:
        raise ValueError(f"n_new_regimes is expected to be greater than n_old_regimes but is {n_new_regimes} < {n_old_regimes}.")
    
    old_regimes = old_regimes.to_numpy() if isinstance(old_regimes, pd.Series) else old_regimes
    new_regimes = new_regimes.to_numpy() if isinstance(new_regimes, pd.Series) else new_regimes
    data = data.to_numpy() if isinstance(data, pd.DataFrame) else data
    
    costs = np.full(
        (n_old_regimes, n_new_regimes),
        fill_value=np.inf,
        dtype=np.float32
    )

    for o in range(n_old_regimes):
        for n in range(n_new_regimes):
            u = data[old_regimes == o]
            v = data[new_regimes == n]
            if len(u) != 0 and len(v) != 0:
                costs[o,n] = get_distance(u, v)

    return costs


def get_regime_map(cost_matrix: np.ndarray) -> np.ndarray:
    """Find the cheapest match of row to columns."""
    cost_matrix = cost_matrix.clip(max=np.finfo(cost_matrix.dtype).max)
    idx = optimize.linear_sum_assignment(cost_matrix)
    return np.stack(idx).T


def add_extra_regime_to_map(map: np.ndarray) -> np.ndarray:
    """
    ```
    array([[0, 2],
           [1, 1],
           [2, 0]])
    ```
    becomes

    ```
    array([[0, 2],
           [1, 1],
           [2, 0],
           [3, 3]])
    ```
    """
    n_regimes_desired = map.shape[0] + 1
    missing_old_regime = list(
        set(np.arange(0, n_regimes_desired)) -
        set(map[:,0])
    )[0]
    missing_new_regime = list(
        set(np.arange(0, n_regimes_desired)) -
        set(map[:,1])
    )[0]
    return np.concatenate([map, [[missing_old_regime, missing_new_regime]]])


def calculate_total_cost(transition_cost_matrix: np.ndarray, norm: float = 1) -> float:
    """Find the best match of old to new regimes and calculate the total cost."""
    row_ind, col_ind = get_regime_map(transition_cost_matrix).T
    cost = transition_cost_matrix[row_ind, col_ind]
    total_cost = cost.sum()
    normalized_cost = total_cost / norm # normalize by the number of old regimes
    return normalized_cost


def added_regime_costs_less(
        regime_cost: float,
        added_regime_cost: float,
    ) -> bool:
    """c0 > c1"""
    return regime_cost > added_regime_cost


def old_regime_is_too_costly(
        regime_cost: float,
        threshold: float,
    ) -> bool:
    """c0 > c-, with c- being a threshold"""
    return regime_cost > threshold


def new_regime_is_advised(
        regime_cost: float,
        added_regime_cost: float,
        threshold: float,
    ) -> bool:
    """Both c0 > c1 and c0 > c-"""
    cond = added_regime_costs_less(
        regime_cost=regime_cost,
        added_regime_cost=added_regime_cost
    )
    cond &= old_regime_is_too_costly(
        regime_cost=regime_cost,
        threshold=threshold,
    )
    return cond


def new_model_collapsed(model_old, model_new, X):
    """New model emits less regimes than old one."""
    return len(np.unique(model_new.predict(X))) < \
        len(np.unique(model_old.predict(X)))

