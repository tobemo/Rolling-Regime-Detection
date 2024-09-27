import json
import os
from collections import deque
from copy import deepcopy
from logging import Logger, getLogger
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score

from base import MyHMM
from my_vhmm import MyVariationalGaussianHMM
from utils import (add_extra_regime_to_map, calculate_total_cost, copy_model,
                   get_regime_map, get_transition_cost_matrix,
                   new_model_collapsed, transfer_model)

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
            n_iter: int = 100,
            tol: float = 1e-6,
            verbose: bool = False,
            name: str = None
        ):
        """If n_components is -1 the ideal number of regimes is auto-detected the first time fit is called, see `initial_fit`.
        If n_components is a list of int then the best number of regimes from that list is detected.
        If n_components is an int >0 then that number of regimes is used."""
        self.name = name or 'root'
        self.models = deque(maxlen=N_REGIME_CLASSIFIERS)
        self._classifier_config = dict(
            n_components=n_components,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
            name=name,
        )
        self._n_components = n_components
        self.scores = []
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_components={self.n_components})"
    
    @property
    def logger(self) -> Logger:
        return getLogger(self.__class__.__name__).getChild(self.name)
    
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
            f"Tracking [{len(self.models)}/{self.models.maxlen}] HMM's."
        )
    
    def __getitem__(self, i: int):
        """Get regime classifier at a previous point in time."""
        if i == -1:
            return self
        
        rc = type(self).from_json(
            self.classifier_to_json()
        )
        i += 1
        for model in list(self.models)[:i]:
            rc.models.append(model)
        return rc

    _classifier_config: dict
    @property
    def classifier_config(self) -> dict:
        """Config used to create RegimeClassifier object."""
        return deepcopy(self._classifier_config)
    @property
    def last_model_config(self) -> dict:
        """The config of the latest model."""
        if self.has_models:
           return self.models[-1].init_config
        return self.classifier_config
    
    @property
    def has_models(self) -> bool:
        return len(self.models) > 0
    
    @property
    def is_fitted(self) -> bool:
        """Last model is fitted."""
        if self.has_models:
            return self.model.is_fitted
        return False
    
    @property
    def n_components(self) -> int:
        """Number of components of last model."""
        if self.has_models:
            return self.model.n_components
        return self._n_components
    
    _deviation_mult = 2
    """How many deviations of the mean the transition cost needs to be before a new regime is added."""
    
    _transition_threshold: float = None
    @property
    def transition_threshold(self) -> float:
        """A new regime should only be added when the transition cost exceeds this threshold.

        Returns n deviations above the mean of past transition costs, as stored as an attribute of each HMM in self.models.
        N depends on class attribute `_deviation_mult`.
        
        Returns inf if not enough data is present.
        """
        if self._transition_threshold:
            return self._transition_threshold
        
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
    @transition_threshold.setter
    def transition_threshold(self, threshold: float) -> float:
        self._transition_threshold = threshold

    def initial_fit(
            self,
            X: np.ndarray | pd.DataFrame,
            lengths: Optional[list[int]]=None,
            k: int = 10
        ) -> None:
        """Train k models and keep best one.
        This is needed to account for gradient descent getting stuck in a local minima. 

        This model will then be the starting point for all future fits.
        
        If n_components in _first_config is -1 the best regime is auto-detected. Only 2-9 are considered for n_components."""
        regimes = self.classifier_config['n_components']
        if regimes == -1:
            regimes = [2, 3, 4, 5, 6, 7, 8, 9]
        elif isinstance(regimes, int):
            regimes = [regimes]
        self.logger.debug(f"Running initial fit with regimes: {regimes}.")

        best_model = None
        best_silhouette_score = -1
        for regime in regimes:
            # first find the best initialization 
            # for the current number of regimes
            cfg = self.classifier_config
            cfg.pop('name')
            cfg['init_params'] = 'stmc'
            cfg['n_components'] = regime
            sub_model = MyVariationalGaussianHMM(**cfg)
            try:
                sub_model.fit(X, lengths=lengths, k=k)
            except Exception as e:
                # raise on last fit only, and only if no model has been fitted
                if not best_model and regime == regimes[-1]:
                    raise e
                else: # just continue
                    self.logger.error(f"Failed to fit {regime} regimes.")
                    self.logger.error(e)
                    continue
            
            # then only keep the best model of all regimes by computing 
            # the silhouette score for the best model in this regime
            y = sub_model.predict(X, lengths=lengths)
            if len(np.unique(y)) < 2:
                continue
            this_silhouette_score = silhouette_score(
                X,
                sub_model.predict(X, lengths=lengths)
            )
            if this_silhouette_score > best_silhouette_score:
                best_model = sub_model
                best_silhouette_score = this_silhouette_score
        
        # ensure a decent model has actually been fitted
        # failure can be each model consistently only detecting one regime
        if not best_model:
            raise RuntimeError('Failed to fit even one working model.')
        
        self.model = best_model
        self.logger.info(
            f"Initial fit found best number of regimes to be {self.model.n_components}"
        )
    
    def _handle_failed_fit(
            self,
            X: np.ndarray | pd.DataFrame,
            lengths: Optional[list[int]]=None
        ) -> None:
        """Logic to handle and track failed fits."""
        # initialize counter if needed and store original value
        if not hasattr(self, '_n_allowed_fails'):
            self._n_allowed_fails = getattr(
                self,
                'fit_can_fail_this_many_times',
                default=6
            )
            self.fit_can_fail_this_many_times = self._n_allowed_fails
        
        # decrease counter on each fail
        self.fit_can_fail_this_many_times -= 1
        if self.fit_can_fail_this_many_times > 0:
            return
    
        self.logger.critical(
            f"Fit failed too many times. Trying to clean slate by running initial_fit."
        )

        # catch initial fit failing
        try:
            self.initial_fit(X, lengths=lengths)
        except Exception as e:
            self.logger.critical("Initial fit failed as well, now stuck with last working model for a while.")
            self.logger.exception(e)
            # reset counter
            self.fit_can_fail_this_many_times = self._n_allowed_fails

    def fit(
            self,
            X: np.ndarray | pd.DataFrame,
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
        if not self.has_models:
            return self.initial_fit(X, lengths=lengths)
        
        # model 1: previous model
        previous_model = copy_model(self.model)

        # model 2: transfer learn with same number of regimes as previous
        new_model = transfer_model(
            old_model=self.model,
            n_iter=max(10, self.models[0].n_iter // 2),
            tol=min(1e-2, self.models[0].tol * 100),
        )
        try:
            new_model.fit(X, lengths=lengths)
        except Exception as e:
            self.logger.warning("Fit failed, reusing previous model.")
            self.logger.exception(e)
            self.model = previous_model
            self._handle_failed_fit(X, lengths=lengths)
            return self

        # model 3: transfer learn but add one extra regime
        new_model_with_added_regime = transfer_model(
            old_model=previous_model,
            n_components=previous_model.n_components + 1,
            n_iter=max(10, self.models[0].n_iter // 2),
            tol=min(1e-2, self.models[0].tol * 100),
        )
        new_model_with_added_regime.fit(X, lengths=lengths)

        # score
        bic_prev = previous_model.bic(X, lengths=lengths)
        bic_new = new_model.bic(X, lengths=lengths)
        bic_new_with_added_regime = new_model_with_added_regime.bic(
            X, lengths=lengths
        )

        # log
        self.logger.debug(
            f"BIC scores are: {bic_prev, bic_new, bic_new_with_added_regime}."
        )
        # determine best model
        if bic_prev < bic_new and bic_prev < bic_new_with_added_regime:
            self.model = previous_model
            self.logger.info("Reusing previous model.")
            return # nothing more is needed
        
        elif bic_new_with_added_regime < bic_new:
            best_model = new_model_with_added_regime
            self.logger.info(
                f"Upped from {self.n_components} to {best_model.n_components} regimes."
            )
        else:
            best_model = new_model
            self.logger.info(
                f"Maintaining {best_model.n_components} regimes."
            )
    
        # check for collapse
        if new_model_collapsed(
            model_new=best_model,
            model_old=previous_model,
            X=X
            ):
            best_model = previous_model
            self.logger.warning(
                f"Newest model is collapsed, reverting back to previous model."
            )
        
        # transition costs from previous labels to new labels
        transition_cost_matrix = get_transition_cost_matrix(
            old_regimes=previous_model.predict(X),
            new_regimes=best_model.predict(X),
            n_old_regimes=previous_model.n_components,
            n_new_regimes=best_model.n_components,
            data=X,
        )

        # mapping of old to new labels
        mapping = get_regime_map(transition_cost_matrix)
        if mapping.shape[0] < best_model.n_components:
            mapping = add_extra_regime_to_map(mapping)
        best_model.mapping = mapping

        # store total cost in model objects themselves
        best_model.transition_cost = calculate_total_cost(
            transition_cost_matrix,
            norm=previous_model.n_components,
        )
                
        # store best model
        self.model = best_model
        return self
    
    def predict(
            self,
            X: np.ndarray | pd.DataFrame,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray | pd.DataFrame:
        """Find most likely state sequence corresponding to X of the last trained model."""
        return self.model.predict(X, lengths=lengths)
    
    def fit_predict(
            self,
            X: np.ndarray | pd.DataFrame,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray | pd.DataFrame:
        """Chains `fit` and `predict`."""
        self.fit(X, lengths=lengths)
        return self.predict(X, lengths=lengths)
    
    def predict_proba(
            self,
            X: np.ndarray | pd.DataFrame,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray | pd.DataFrame:
        """Compute the posterior probability for each state of the last trained model."""
        return self.model.predict_proba(X, lengths=lengths)
    
    def predict_all(
            self,
            X: np.ndarray | pd.DataFrame,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray | pd.DataFrame:
        """row: time, col: model"""
        predictions = [model.predict(X) for model in self.models]
        if isinstance(X, np.ndarray):
            return np.stack(predictions).T
        elif not isinstance(X.index, pd.DatetimeIndex):
            return pd.DataFrame(np.stack(predictions).T)
        
        predictions = {i: p for i, p in enumerate(predictions)}
        predictions = pd.DataFrame(predictions)
        predictions.columns.name = 'Model'

        ends = [
            pd.Timestamp(model.timestamp, unit='s', tz=X.index.tz)
            for model in self.models
        ]
        ends = X.index.get_indexer(ends)
        assert ends.min() >= 0
        ends += 1
        for i, end in enumerate(ends):
            predictions.iloc[end:, i] = None

        return predictions
    
    @classmethod
    def from_json(cls, config: str):
        # create regime classifier object
        config = json.loads(config)
        regime_classifier = cls(**config)
        return regime_classifier
    
    def add_models_from_json(self, configs: list[str]) -> None:
        # load models and sort by timestamp
        configs = configs[-self.models.maxlen:]
        models = [MyVariationalGaussianHMM.from_json(cfg) for cfg in configs]
        models.sort(key=lambda m: m.timestamp)

        # add to regime classifier
        for model in models:
            self.models.append(model)

    @classmethod
    def from_jsons(cls, classifier_config: str, configs: list[str]):
        """Fill `self.models` with MyVariationalGaussianHMM initialized using a json string."""
        # create regime classifier object
        regime_classifier = cls.from_json(classifier_config)

        # load models and sort by timestamp
        configs = configs[-regime_classifier.models.maxlen:]
        models = [MyVariationalGaussianHMM.from_json(cfg) for cfg in configs]
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
        """Returns initial kwargs, used to initialize this RegimeClassifier object, as a json string."""
        return json.dumps(self.classifier_config)
    
    def models_to_jsons(self) -> list[str]:
        """Returns a dict of all tracked models as jsons. Keys are the creation times of each model."""
        return [model.to_json() for model in self.models]
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, RegimeClassifier):
            return False
        
        if not self.is_fitted and not other.is_fitted:
            return self.classifier_config == other.classifier_config
        
        if self.has_models and other.has_models:
            cond_a = self.model == other.model
            cond_b = self.transition_threshold == other.transition_threshold
            return cond_a & cond_b
        
        return False

    def scatter(
        self,
        X: np.ndarray | pd.DataFrame,
        lengths: Optional[list[int]]=None
    ) -> plt.Axes:
        ax = self.model.scatter(X, lengths=lengths)
        plt.figtext(
            0.99, 0.01,
            'Using hindsight bias!',
            horizontalalignment='right'
        )
        return ax

