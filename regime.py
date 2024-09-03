import json
import os
from collections import deque
from copy import deepcopy
from logging import Logger, getLogger
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.metrics import silhouette_score

from base import MyHMM


N_REGIME_CLASSIFIERS = int(os.getenv('MAX_REGIME_CLASSIFIERS', 128))


class RegimeClassifier():
    """A wrapper class that fits and keeps track of hidden Markov models over time. At each fitting it checks whether it makes sense to add one regime or not. This to try to capture the emerge of new regimes over time.
    The label assignment of each new model is matched with the preceding model to ensure a continuous labeling. Otherwise there is no guarantee that model_t and model_t-1 will use the same label when talking about the same regime.

    Inspired by "Robust rolling regime detection" by Hirsa et al.
    
    Operation:
    At first 10 models are trained and the best one is kept.
    From then on new models are transferred from this one. The previous transition matrix and starting probabilities are used to initialize a new model which then is fitted on the most recent up-to-date data. Data is treated as an expanding window, meaning that as more data comes available, new models are trained using all available data up to that point.

    Some terminology:
    transition cost: The distance between the distributions of the regimes at t_n and at t_n-1. Minimized to match labels from regimes_n-1 with those of regimes_n

    At each new fit both a model with the same number of regimes as last time (K), as well as a model with 1 more regime (K+1), are fitted. The one with an extra regime, K+1, is used if it is both cheaper than K, and, the K one has a cost that exceeds a certain threshold (in this implementation μ+n*σ is used, see `transition_threshold`).
    - """
    def __init__(
            self,
            n_components: int | list[int] = -1,
            n_iter: int =100,
            tol: float =1e-6,
            random_state: float = None,
            verbose: bool = False,
            name: str = None
        ):
        """If n_components is -1 the ideal number of regimes is auto-detected the first time fit is called, see `initial_fit`."""
        self.name = 'root' or name
        self._logger = getLogger(self.__class__.__name__)
        self.models = deque(maxlen=N_REGIME_CLASSIFIERS)
        self._first_config = dict(
            n_components=n_components,
            init_params="stmc",
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
        )
        self._n_components = n_components
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_components={self.n_components})"
    
    @property
    def logger(self) -> Logger:
        return self._logger.getChild(self.name)
    
    models: deque[MyHMM]
    """List of all trained models."""
    @property
    def model(self) -> MyHMM:
        """Last trained model."""
        if not self.has_models:
            raise AttributeError("RegimeClassifier currently tracks 0 models.")
        return self.models[-1]
    @model.setter
    def model(self, model) -> None:
        """Store new model into model list."""
        self.models.append(model)
        self.logger.info(
            f"Tracking {len(self.models)}/{self.models.maxlen} HMM's."
        )
    
    def __getitem__(self, i: int) -> MyHMM:
        return self.models[i]
    
    _first_config: dict
    @property
    def config(self) -> dict:
        """The config of the latest model."""
        return self.models[-1].config
    
    @property
    def has_models(self) -> bool:
        return len(self.models) > 0
    
    @property
    def is_fitted(self) -> bool:
        if self.has_models:
            return self.model.is_fitted
        return False
    
    @property
    def n_components(self) -> int:
        if self.has_models:
            return self.model.n_components
        return self._n_components
    
    _deviation_mult = 2
    """How many deviations of the mean the transition cost needs to be before a new regime is added."""
    
    @property
    def transition_threshold(self) -> float:
        """A new regime should only be added when the transition cost exceeds this threshold.

        Returns n deviations above the mean of past transition costs, as stored as an attribute of each HMM in self.models.
        N depends on class attribute `_deviation_mult`.
        
        Returns inf if not enough data is present.
        """
        costs = [model.transition_cost for model in self.models]
        costs = np.array(costs)
        costs = costs[costs < np.inf]
        
        if len(costs) == 0:
            return np.inf
        if len(costs) < 8:
            return 1.2 * costs.max()
        
        mean = costs.mean()
        std = costs.std()
        return mean + self._deviation_mult * std

    def initial_fit(
            self,
            X: np.ndarray,
            lengths: Optional[list[int]]=None,
            k: int = 10
        ) -> None:
        """Train k models and keep best one.
        This is needed to account for gradient descent getting stuck in a local minima. 

        This model will then be the starting point for all future fits.
        
        If n_components in _first_config is -1 the best regime is auto-detected. Only 2-9 are considered for n_components."""
        regimes = self._first_config['n_components']
        if regimes == -1:
            regimes = [2, 3, 4, 5, 6, 7, 8, 9]
        else:
            regimes = [regimes]

        best_model = None
        best_silhouette_score = -1
        for regime in regimes:
            # first find the best initialization for the current regime
            best_sub_model = None
            best_log_likelihood = -np.inf
            for _ in range(k):
                this_model = MyHMM(self._first_config)
                this_model.fit(X, lengths=lengths)
                this_log_likelihood = this_model.score(X)
                if this_log_likelihood > best_log_likelihood:
                    best_sub_model = this_model
                    best_log_likelihood = this_log_likelihood
            
            # then only keep the best model of all regimes by computing 
            # the silhouette score for the best model in this regime
            this_silhouette_score = silhouette_score(
                X,
                best_sub_model.predict(X, lengths=lengths)
            )
            if this_silhouette_score > best_silhouette_score:
                best_model = best_sub_model
                best_silhouette_score = this_silhouette_score
        
        self.model = best_model

    def fit(
            self,
            X: np.ndarray,
            lengths: Optional[list[int]]=None
        ) -> None:
        """Fit two new regime models, one with the same amount of regimes and one with one more. The one that has the cheapest transition costs, compared to the previous regime classifier, is used.

        This way a chronological chain of models emerges, each transferred from the previous, with the origin being whatever model was deemed best in `initial_fit`.
        """
        # compare model_n_t to model_n_t-1
        # compare model_n+1_t to model_n_t-1
        # with n the number of regimes and t the time point
        # get cost of both
        # check if model_n+1_t costs less than model_n_t,
        #  AND model_n_t is costlier than a certain threshold (indicating that for the current data at t it fails to effectively capture all regimes)
        
        # call initial fit if no models exist
        if len(self.models) == 0:
            return self.initial_fit(X, lengths=lengths)
        
        # model 1: previous model
        model_previous = self.model

        # model 2: transfer learn with same number of regimes as previous
        new_model = copy_model(
            old_model=self.model,
            config=self.config,
        )
        new_model.fit(X, lengths=lengths)

        # model 3: transfer learn but add one extra regime
        config = deepcopy(self.config)
        config['n_components'] += 1
        new_model_with_added_regime = copy_model_and_add_regimes(
            old_model=model_previous,
            config=config,
        )
        new_model_with_added_regime.fit(X, lengths=lengths)

        # regimes for all three models on all known data
        regimes_previous = model_previous.predict(X)
        regimes_new = new_model.predict(X)
        regimes_new_added_regime = new_model_with_added_regime.predict(X)

        # transition costs for both new models
        transition_cost_matrix_current_regimes = get_transition_cost_matrix(
            old_regimes=regimes_previous,
            new_regimes=regimes_new,
            n_old_regimes=model_previous.n_components,
            n_new_regimes=new_model.n_components,
            data=X,
        )
        transition_cost_matrix_added_regime = get_transition_cost_matrix(
            old_regimes=regimes_previous,
            new_regimes=regimes_new_added_regime,
            n_old_regimes=model_previous.n_components,
            n_new_regimes=new_model_with_added_regime.n_components,
            data=X,
        )

        # store costs in model objects themselves
        new_model.transition_cost = calculate_total_cost(
            transition_cost_matrix_current_regimes
        )
        new_model_with_added_regime.transition_cost = calculate_total_cost(
            transition_cost_matrix_added_regime
        )

        # store mapping from previous to current regime 
        # in model objects themselves
        new_model.mapping = match_regimes(
            transition_cost_matrix_current_regimes
        )
        new_model_with_added_regime.mapping = match_regimes(
            transition_cost_matrix_added_regime
        )

        # logging
        self.logger.debug(f"Cost as is: {new_model.transition_cost}")
        self.logger.debug(
            f"Cost with extra regime: {new_model_with_added_regime.transition_cost}"
        )
        self.logger.debug(f"Transition threshold: {self.transition_threshold}")
        
        # compare costs
        cheapest_model = new_model
        if new_regime_is_advised(
            new_regime_cost=transition_cost_matrix_added_regime,
            old_regime_cost=transition_cost_matrix_current_regimes,
            threshold=self.transition_threshold,
        ):
            cheapest_model = new_model_with_added_regime
            self.logger.info(
                f"Upped from {self.n_components} to {cheapest_model.n_components} regimes."
            )
                
        # store cheapest model
        self.model = cheapest_model
    
    def predict(
            self,
            X: np.ndarray,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray:
        """Find most likely state sequence corresponding to X of the last trained model."""
        self.model.predict(X, lengths=lengths)
    
    def predict_proba(
            self,
            X: np.ndarray,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray:
        """Compute the posterior probability for each state of the last trained model."""
        self.model.predict_proba(X, lengths=lengths)

    @classmethod
    def from_jsons(cls, meta_config: str, configs: list[str]):
        """Fill `self.models` with MyVariationalGaussianHMM initialized using a json string."""
        if len(configs) == 0:
            raise RuntimeError(
                "Can't initialize 'RegimeClassifier' since configs is an empty list."
            )
        
        # create regime classifier object
        meta_config = json.loads(meta_config)
        regime_classifier = cls(**meta_config)

        # load models and sort by timestamp
        configs = configs[-regime_classifier.models.maxlen:]
        models = [MyHMM.from_json(cfg) for cfg in configs]
        models.sort(key=lambda m: m.timestamp)

        # add to regime classifier
        for model in models:
            regime_classifier.models.append(model)
        
        return regime_classifier
    
    def to_json(self) -> tuple[str, dict[int, str]]:
        """Returns a dict of initial kwargs for this class and all the jsons that describe the models currently being tracked.
        
        See `classifier_to_json` and `models_to_jsons`."""
        return self.classifier_to_json(), self.models_to_jsons()
    
    def classifier_to_json(self) -> str:
        """Returns initial kwargs, used to initialize this RegimeClassifier object, as json string."""
        return json.load(self._first_config)
    
    def models_to_jsons(self) -> dict[int, str]:
        """Returns a dict of all tracked models as jsons. Keys are the creation times of each model."""
        return {model.timestamp: model.to_json() for model in self.models}


def copy_model(
        old_model: MyHMM,
        config: dict,
        ) -> MyHMM:
    """Return a new model of the same type as old model with its transition matrix and start probabilities copied over."""
    config['init_params'] = 'mc'
    new_model = MyHMM(config)
    new_model.transmat_ = old_model.transmat_
    new_model.startprob_ = old_model.startprob_
    return new_model


def copy_model_and_add_regimes(
        old_model: MyHMM,
        config: dict,
        ) -> MyHMM:
    """Same as `copy_model` but one or more regimes are added. The difference between n_components in dict and in old_model determines how many regimes are added."""
    n_components = old_model.n_components
    old_startprob = old_model.startprob_
    old_transmat = old_model.transmat_
    old_means = old_model.means_
    old_covars = old_model.covars_

    # only estimate mean and covariance before fit
    config['init_params'] = 'mc'
    new_model = MyHMM(config)
    
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
    """Returns cost matrix.
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


def match_regimes(transition_cost_matrix: np.ndarray) -> np.ndarray:
    """Find the best match of old to new regimes."""
    idx = optimize.linear_sum_assignment(transition_cost_matrix)
    return np.stack(idx).T


def calculate_total_cost(transition_cost_matrix: pd.DataFrame) -> float:
    """Find the best match of old to new regimes and calculate the total cost."""
    transition_cost_matrix = transition_cost_matrix.to_numpy()
    row_ind, col_ind = match_regimes(transition_cost_matrix).T
    total_cost = transition_cost_matrix[row_ind, col_ind].sum()
    normalized_cost = total_cost / len(row_ind)  # Normalize by the number of old regimes
    return normalized_cost


def new_regime_costs_less(
        old_regime_cost: float,
        new_regime_cost: float,
    ) -> bool:
    """c0 > c1"""
    return old_regime_cost > new_regime_cost


def old_regime_is_too_costly(
        old_regime_cost: float,
        threshold: float,
    ) -> bool:
    """c0 > c-, with c- being a threshold"""
    return old_regime_cost > threshold


def new_regime_is_advised(
        old_regime_cost: float,
        new_regime_cost: float,
        threshold: float,
    ) -> bool:
    """Both c0 > c1 and c0 > c-"""
    cond = new_regime_costs_less(
        old_regime_cost=old_regime_cost,
        new_regime_cost=new_regime_cost
    )
    cond &= old_regime_is_too_costly(
        old_regime_cost=old_regime_cost,
        threshold=threshold,
    )
    return cond

