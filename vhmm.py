import json
from copy import deepcopy

import numpy as np
import pandas as pd
from hmmlearn import vhmm
from scipy import optimize, stats


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
    
    @property
    def config(self) -> dict:
        return {
            "n_components": self.n_components,
            "init_params": self.init_params,
            "random_state": self.random_state,
            "n_iter": self.random_state,
            "tol": self.tol,
            "verbose": self.verbose,
        }
    
    def __init__(
            self,
            n_components=1,
            init_params='stmc',
            random_state=None,
            n_iter=100,
            tol=1e-6,
            verbose=False,
        ) -> None:
        super().__init__(
            n_components=n_components, 
            covariance_type="full",
            algorithm="viterbi",
            init_params=init_params,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            params='stmc',
            implementation='scaling',
        )
    
    def to_json(self) -> str:
        config = {
            "n_components": self.n_components,
            "init_params": self.init_params,
            "random_state": self.random_state,
            "n_iter": self.random_state,
            "tol": self.tol,
            "verbose": self.verbose,
            "covariance_type": self.covariance_type,
            "startprob": self.startprob_.tolist(),
            "transmat": self.transmat_.tolist(),
            "means": self.means_.tolist(),
            "covars": self.covars_.tolist(),
            "weights": self.weights_.tolist(),
            "precisions_cholesky": self.precisions_cholesky_.tolist(),
            "dirichlet_concentration_prior": self.dirichlet_concentration_prior_.tolist(),
            "mean_precision_prior": self.mean_precision_prior_.tolist(),
            "degrees_of_freedom_prior": self.degrees_of_freedom_prior_.tolist(),
        }
        if self.is_fitted:
            config["init_params"] = ""
        return json.dumps(config)

    @staticmethod
    def from_json(config: str):
        # re-init
        config = json.loads(config)
        model = MyVariationalGaussianHMM.__init__(
            n_components=config["n_components"],
            init_params=config["init_params"],
            random_state=config["random_state"],
            n_iter=config["n_iter"],
            tol=config["tol"],
            verbose=config["verbose"],
        )
        
        # assign parameters
        model.startprob_ = np.array(config["startprob"])
        model.transmat_ = np.array(config["transmat"])
        model.means_ = np.array(config["means"])
        model.covars_ = np.array(config["covars"])
        model.weights_ = np.array(config["weights"])
        model.precisions_cholesky_ = np.array(config["precisions_cholesky"])
        model.dirichlet_concentration_prior_ = np.array(
            config["dirichlet_concentration_prior"]
        )
        model.mean_precision_prior_ = np.array(
            config["mean_precision_prior"]
        )
        model.degrees_of_freedom_prior_ = np.array(
            config["degrees_of_freedom_prior"]
        )

        return model


class RegimeClassifier():
    models: list[MyVariationalGaussianHMM] = []
    """List of all trained models."""
    @property
    def model(self) -> MyVariationalGaussianHMM:
        """Last trained model."""
        return self.models[-1]
    @model.setter
    def model(self, model) -> None:
        """Store new model into model list."""
        self.models.append(model)
    
    _first_config: dict
    @property
    def first_config(self) -> dict:
        if len(self.models) > 0:
            return self.models[0].config
        else: return self._first_config
    @first_config.setter
    def first_config(self, config) -> None:
        self._first_config = config
    
    @property
    def config(self) -> dict:
        return self.models[-1].config

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
            model_ = MyVariationalGaussianHMM(self.first_config)
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
        
        # call initial fit instead if no models exist
        if len(self.models) == 0:
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
        
        # store cheapest model
        self.model = cheapest_model
    
    def predict(self, X, lengths=None) -> None:
        self.model.predict(X, lengths=lengths)
    
    def predict_proba(self, X, lengths=None) -> None:
        self.model.predict_proba(X, lengths=lengths)

    def __init__(
            self,
            n_components=1,
            n_iter=100,
            tol=1e-6,
            random_state=None,
            verbose=False,
        ):
        self.first_config = dict(
            n_components=n_components,
            init_params="stmc",
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
        )

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
    new_model = MyVariationalGaussianHMM(config)
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
    new_model = MyVariationalGaussianHMM(config)
    
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
