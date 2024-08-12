import numpy as np
from hmmlearn import vhmm
import pandas as pd
from scipy import stats, optimize
from copy import deepcopy


class MyVariationalGaussianHMM(vhmm.VariationalGaussianHMM):
    """VariationalGaussianHMM with some settings fixed:
    
    covariance_type: 'full'
    implementation: 'scaling'
    params: 'stmc'

    # s: start probability
    # t: transition matrix
    # m: mean
    # c: covariance matrix
    """
    @property
    def is_fitted(self) -> bool:
        return hasattr(self, 'transmat_')
    
    def __init__(
            self,
            n_components=1,
            algorithm="viterbi",
            init_params='stmc',
            random_state=None,
            n_iter=100,
            tol=1e-6,
            verbose=False,
            **kwargs,
        ) -> None:
        super().__init__(
            n_components=n_components, 
            covariance_type="full",
            algorithm=algorithm,
            init_params=init_params,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params='stmc',
            implementation='scaling',
            **kwargs,
        )


class RegimeClassifier():
    # The HMM is completely determined by start probability vector Pi, transition probability matrix A, and emission probability theta
    model: MyVariationalGaussianHMM
    models: list[MyVariationalGaussianHMM]
    first_config: dict
    config: dict

    def _store_snapshot(self) -> None:
        """Store model and parameters startprob_, transmat_, means_ and covars_.
        Called after every fit."""
        self.models.append(self.model)
        self.start_probabilities.append(self.model.startprob_)
        self.transition_matrices.append(self.model.transmat_)
        self.means.append(self.model.means_)
        self.covariance_matrices.append(self.model.covars_)

    def initial_fit(self, X, lengths=None, k: int = 10) -> None:
        """Train k models and keep best one."""
        model = self.init_model(**self.config)
        model.fit(X, lengths=lengths)
        score = model.score(X)

        for _ in range(k-1):
            model_ = self.init_model(**self.config)
            model_.fit(X, lengths=lengths)
            score_ = model_.score(X)
            if score_ > score:
                model = model_
                score = score_
        
        self.model = model
        self._store_snapshot()

    def fit(self, X, lengths=None) -> None:
        if not hasattr(self, 'model_'):
            return self.initial_fit(X, lengths=lengths)
        
        # This should train one model with n_components and one with n_components+1; and keep the best one

        # copy and fit model with same number of regimes as current
        # transfer learn from previous model
        model_transferred = copy_model(
            old_model=self.model,
            config=self.config,
        )
        model_transferred.fit(X, lengths=lengths)

        # copy and fit model but add one extra regime
        # transfer learn from newest model
        config = deepcopy(self.config)
        config['n_components'] += 1
        model_transferred_w_added_regime = copy_model_and_add_regimes(
            old_model=model_transferred,
            config=config,
        )
        model_transferred_w_added_regime.fit(X, lengths=lengths)
        
        # TODO: costs
        self.model = model_transferred
        old_model_is_more_costly = False
        
        if old_model_is_more_costly:
            self.model = model_transferred_w_added_regime
            self.config = config
        self._store_snapshot()
    
    def predict(self, X, lengths=None) -> None:
        self.model.predict(X, lengths=lengths)
    
    def predict_proba(self, X, lengths=None) -> None:
        self.model.predict_proba(X, lengths=lengths)

    def __init__(
            self,
            n_components=1,
            algorithm="viterbi",
            init_params='stmc',
            random_state=None,
            n_iter=100,
            tol=1e-6,
            verbose=False,
            **kwargs,
        ):
        config = dict(
            n_components=n_components, 
            algorithm=algorithm,
            init_params=init_params,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            **kwargs,
        )
        self.first_config = deepcopy(config)
        self.config = config

        self.start_probabilities = []
        self.transition_matrices = []
        self.means = []
        self.covariance_matrices = []

    @property
    def is_fitted(self) -> bool:
        return self.model.is_fitted
    
    @property
    def n_components(self) -> int:
        return self.model.n_components


def copy_model(
        old_model: MyVariationalGaussianHMM,
        config: dict,
        ) -> MyVariationalGaussianHMM:
    config['init_params'] = 'mc'
    new_model = MyVariationalGaussianHMM.init_model(config)
    new_model.transmat_ = old_model.transmat_
    new_model.covars_ = old_model.covars_
    return new_model


def copy_model_and_add_regimes(
        old_model: MyVariationalGaussianHMM,
        config: dict,
        ) -> MyVariationalGaussianHMM:
    n_components = old_model.n_components
    old_startprob = old_model.startprob_
    old_transmat = old_model.transmat_
    old_means = old_model.means_
    old_covars = old_model.covars_

    # only estimate mean and covariance before fit
    config['init_params'] = 'mc'
    new_model = MyVariationalGaussianHMM.init_model(config)
    
    # regimes to add
    n = config['n_components'] - n_components

    # the new start probability is set to the smallest one already present
    new_startprob = np.zeros(n_components + n)
    new_startprob[:-n] = old_startprob
    new_startprob[-n:] = old_startprob.min()
    new_startprob /= new_startprob.sum()
    new_model.startprob_ = new_startprob

    # for a new regime the transitions are set to be equally probable to transition to any other existing regime
    # for existing regimes to switch to the new ones, it is assumed that this is as unlikely as the least likely transition already present
    new_transmat = np.zeros(
        (n_components + n, n_components + n)
    )
    new_transmat[:n_components, :n_components] = old_transmat
    new_transmat[-n:, :] = 1 
    new_transmat[:, -n:] = old_transmat.min(axis=1)
    new_transmat /= new_transmat.sum(axis=1, keepdims=True)
    new_model.transmat_ = new_transmat

    # # the mean(s) of the new hidden state are initialized to 0
    # new_means = np.zeros((n_components + n, *old_means.shape[1:]))
    # new_means[:-n] = self.means_
    # new_means[-n:] = 0
    # new_model.means_ = new_means

    # new_covars = np.zeros((n_components + n, *old_covars.shape[1:]))
    # new_covars[:-n, :, :] = old_covars
    # new_covars[-n, :, :] = np.eye(old_covars.shape[-1]) + self.min_covar
    # new_model.covars_ = new_covars

    return new_model


def get_distance(u: np.ndarray, v: np.ndarray) -> float:
    if u.ndim == 2:
        return np.mean(
            [stats.wasserstein_distance(u[i], v[i]) for i in range(u.shape[1])]
        )
    return stats.wasserstein_distance(u, v)


def get_transition_cost_matrix(
        old_regimes: pd.Series,
        new_regimes: pd.Series,
        n_old_regimes: int,
        n_new_regimes: int,
        old_data: pd.Series,
        new_data: pd.Series,
        fill_value: float = 0.
    ) -> pd.DataFrame:
    """Returns normalized cost matrix.
    The cost matrix describes the distance between each old and new distribution."""
    I = n_old_regimes
    J = n_new_regimes
    if J < I:
        raise ValueError(f"J is expected to be ge than I but is {J} < {I}.")

    costs = {}
    for i in range(I):
        for j in range(J):
            u = old_data[old_regimes == i].to_numpy()
            v = new_data[new_regimes == j].to_numpy()
            if len(u) == 0 or len(v) == 0:
                costs[(i,j)] = np.inf
                continue
            costs[(i,j)] = get_distance(u, v)
    costs = pd.Series(costs)
    costs.index.names = ['old/I', 'new/J']
    costs = costs.unstack(level=-1)
    if J > I:
        costs = costs.reindex(np.arange(J), fill_value=fill_value) # ensure squareness
    return costs


def match_regimes(transition_cost_matrix: np.ndarray) -> tuple:
    return optimize.linear_sum_assignment(transition_cost_matrix)


def calculate_total_cost(transition_cost_matrix: pd.DataFrame) -> float:
    transition_cost_matrix = transition_cost_matrix.to_numpy()
    row_ind, col_ind = match_regimes(transition_cost_matrix)
    total_cost = transition_cost_matrix[row_ind, col_ind].sum()
    normalized_cost = total_cost / len(row_ind)  # Normalize by the number of old regimes
    return normalized_cost


def new_regime_costs_less(
        old_regime_cost: float,
        new_regime_cost: float,
    ) -> bool:
    return old_regime_cost > new_regime_cost

def old_regime_is_too_costly(
        old_regime_cost: float,
        threshold: float,
    ) -> bool:
    return old_regime_cost > threshold

def new_regime_is_advised(
        old_regime_cost: float,
        new_regime_cost: float,
        threshold: float,
    ) -> bool:
    cond = new_regime_costs_less(
        old_regime_cost=old_regime_cost,
        new_regime_cost=new_regime_cost
    )
    cond &= old_regime_is_too_costly(
        old_regime_cost=old_regime_cost,
        threshold=threshold,
    )
    return cond