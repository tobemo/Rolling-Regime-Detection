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
    
    _transition_cost = np.inf
    @property
    def transition_cost(self) -> float:
        return  self._transition_cost
    @transition_cost.setter
    def transition_cost(self, cost: float) -> None:
        self._transition_cost = cost
    
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
    _model: MyVariationalGaussianHMM
    models: list[MyVariationalGaussianHMM]
    first_config: dict
    config: dict

    @property
    def model(self) -> MyVariationalGaussianHMM:
        return self._model
    @model.setter
    def model(self, model) -> None:
        self.models.append(model)
        self._model = model
    
    @property
    def transition_threshold(self) -> float:
        """2 deviations above the mean of past transition costs.
        Returns inf if not enough data is present."""
        costs = [model.transition_cost for model in self.models]
        costs = np.array(costs)
        costs = costs[costs < np.inf]
        if len(costs) < 8:
            return np.inf
        mean = costs.mean()
        std = costs.std()
        return mean + 2 * std

    def initial_fit(self, X, lengths=None, k: int = 10) -> None:
        """Train k models and keep best one.
        This is needed to account for gradient descent getting stuck in a local minima."""
        model = None
        score = -np.inf

        for _ in range(k):
            model_ = MyVariationalGaussianHMM.init_model(self.config)
            model_.fit(X, lengths=lengths)
            score_ = model_.score(X)
            if score_ > score:
                model = model_
                score = score_
        
        self.model = model

    def fit(self, X, lengths=None) -> None:
        """Fit two new regime models, one with the same amount of regimes and one with one more. The one that has the cheapest transition costs is kept.
        """
        # compare model_n_t to model_n_t-1
        # compare model_n+1_t to model_n_t-1
        # get cost of both
        # check if model_n+1_t costs less than model_n_t,
        #  AND model_n_t is costlier than a certain threshold (indicating that for the current data at t it fails to effectively capture all regimes)

        if not hasattr(self, '_model'):
            return self.initial_fit(X, lengths=lengths)
        
        # model 1: previous model
        model_previous = self.model

        # model 2: transfer learn with same number of regimes as previous
        model_new = copy_model(
            old_model=self.model,
            config=self.config,
        )
        model_new.fit(X, lengths=lengths)

        # model 3: transfer learn but add one extra regime
        config = deepcopy(self.config)
        config['n_components'] += 1
        model_new_added_regime = copy_model_and_add_regimes(
            old_model=model_previous,
            config=config,
        )
        model_new_added_regime.fit(X, lengths=lengths)

        # regimes for all three models
        regimes_previous = model_previous.predict(X)
        regimes_new = model_new.predict(X)
        regimes_new_added_regime = model_new_added_regime.predict(X)

        # transition costs for both new models
        transition_cost_current_regimes = get_transition_cost_matrix(
            old_regimes=regimes_previous,
            new_regimes=regimes_new,
            n_old_regimes=model_previous.n_components,
            n_new_regimes=model_new.n_components,
            data=X,
        )
        transition_cost_added_regime = get_transition_cost_matrix(
            old_regimes=regimes_previous,
            new_regimes=regimes_new_added_regime,
            n_old_regimes=model_previous.n_components,
            n_new_regimes=model_new_added_regime.n_components,
            data=X,
        )
        model_new.transition_cost = transition_cost_current_regimes
        model_new_added_regime.transition_cost = transition_cost_added_regime
        
        # compare costs
        cheapest_model = model_new
        if new_regime_is_advised(
            new_regime_cost=transition_cost_added_regime,
            old_regime_cost=transition_cost_current_regimes,
            threshold=self.transition_threshold,
        ):
            cheapest_model = model_new_added_regime
            self.config = config # update config
        
        # store cheapest model
        self.model = cheapest_model
    
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

        self.models = []

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
    """The difference between n_components in dict and in old_model determines how many regimes are added."""
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
        old_regimes: np.ndarray,
        new_regimes: np.ndarray,
        n_old_regimes: int,
        n_new_regimes: int,
        data: pd.Series,
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
            u = data[old_regimes == i]
            v = data[new_regimes == j]
            if len(u) == 0 or len(v) == 0:
                costs[(i,j)] = np.inf
            else:
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


def compare_models(
        model_a: MyVariationalGaussianHMM,
        model_b: MyVariationalGaussianHMM,
        X: np.ndarray,
    ):
    regimes_a = model_a.predict(X)
