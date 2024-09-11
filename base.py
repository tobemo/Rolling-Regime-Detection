from copy import deepcopy
import json
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from hmmlearn import hmm


def _validate_mapping(mapping: np.ndarray, n_components: int) -> None:
    if mapping.shape != (n_components, 2):
        raise ValueError(
            f"Mapping should be of shape {(n_components, 2)}, but is of shape {mapping.shape}"
        )
    wrong_mapping = not np.array_equal(
        np.sort(mapping[:,0]),
        np.arange(n_components)
    )
    wrong_mapping |= not np.array_equal(
        np.sort(mapping[:,1]),
        np.arange(n_components)
    )
    if wrong_mapping:
        raise ValueError(
            f"Wrong values in mapping, expected elements of {np.arange(n_components)} in both 1st and 2nd column, but got {mapping}."
        )


class MyHMM(ABC):
    @property
    def is_fitted(self) -> bool:
        return hasattr(self, 'transmat_')
    
    _transition_cost: float
    @property
    def transition_cost(self) -> float:
        """Transition cost from a unspecified model to this one. Assumes HMMs are use in a chain so the transition cost from the previous model to this one is known."""
        return self._transition_cost
    @transition_cost.setter
    def transition_cost(self, cost: float) -> None:
        self._transition_cost = cost

    timestamp: int
    """Fit time."""

    _mapping: np.ndarray
    @property
    def mapping(self) -> np.ndarray:
        """A 2D array where the first column corresponds to the from and the second column corresponds to the to. Should be of shape (n_components, 2)."""
        return getattr(
            self,
            '_mapping',
            np.repeat(
                np.arange(self.n_components)[:,None], 2, axis=1
            )
        )
    @mapping.setter
    def mapping(self, m: np.ndarray) -> None:
        _validate_mapping(m, self.n_components)
        self._mapping = m
    @property
    def mapper(self) -> dict:
        """A mapping dict where keys are the from and values are the to.
        """
        # turn 2 array into dict with first col being keys
        # and second col being values
        return dict(
            zip(
                *self.mapping.T.tolist()
            )
        )
    
    @property
    @abstractmethod
    def HMM(self):
        """HMM model to use."""
        pass
    
    @property
    @abstractmethod
    def HMM_config(self):
        """Config for HMM model."""
        pass
    
    def __init__(self, timestamp: int = None) -> None:
        super().__init__()
        self.timestamp = timestamp or int(time.time())
        self._transition_cost = np.inf
    
    def _multi_fit(self,
            X: np.ndarray,
            lengths: Optional[list[int]] = None,
            k: int = 5,
        ):
        score = -np.inf
        model = None
        config = self.HMM_config
        for _ in range(k):
            config['random_state'] = np.random.randint(1e6)
            _model = self.HMM(**config)
            _model.fit(X, lengths)
            _score = _model.score(X)
            if _score > score:
                model = _model
                score = _score
        
        model = model or _model
        self.__dict__.update(model.__dict__)
        return self
        
    def fit(
            self,
            X: np.ndarray,
            lengths: Optional[list[int]] = None,
            k: int = 5,
        ):
        self.timestamp = int(time.time())
        if self.random_state:
            return super().fit(X, lengths)
        return self._multi_fit(X, lengths, k)
    
    def map_predictions(self, predictions: np.ndarray) -> np.ndarray:
        return np.select(
            [predictions == i for i in self.mapping[:,1]],
            self.mapping[:,0]
        )
    
    def _predict(
            self,
            X: np.ndarray,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray:
        """Find most likely state sequence corresponding to ``X``. Without mapping."""
        assert self.is_fitted, "Model is not fitted."
        return super().predict(X, lengths=lengths)

    def predict(
            self,
            X: np.ndarray,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray:
        """Find most likely state sequence corresponding to ``X``. States are mapped using self.mapper."""
        predictions = self._predict(X, lengths)
        predictions = self.map_predictions(predictions)
        return predictions
    
    def predict_proba(
            self,
            X: np.ndarray,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray:
        if hasattr(self, '_mapping'):
            raise NotImplementedError("Using predict_proba while having a mapper set is not supported.")
        assert self.is_fitted, "Model is not fitted."
        probas = super().predict_proba(X, lengths=lengths)
        reordering = self.mapping[:,1]
        return probas[:, reordering]

    @property
    def init_config(self) -> dict:
        """Returns creation config."""
        return deepcopy({
            "n_components": self.n_components,
            "init_params": self.init_params,
            "random_state": self.random_state,
            "n_iter": self.n_iter,
            "tol": self.tol,
            "verbose": self.verbose,
            "timestamp": self.timestamp
        })

    def get_config(self) -> dict:
        """Get config that exactly describes this model.
        This not only includes thins like creation time and number of components but also things like start probabilities.
        
        Can be used to initialize this class to an already fitted, and ready to use, model; when using MyVariationalGaussianHMM.from_config()
        
        This enables serialization and persistence."""
        config = {
            "timestamp": self.timestamp,
            "transition_cost": self.transition_cost
        }
        if hasattr(self, "mapping"):
            config["mapping"] = self.mapping.tolist()
        return config

    def to_json(self) -> str:
        """Model to json string."""
        config = self.get_config()
        return json.dumps(config)

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict):
        pass
        
    @classmethod
    def from_json(cls, config: str):
        """Model from json string."""
        config = json.loads(config)
        return cls.from_config(config)

    def _check(self) -> None:
        # Don't call check on fitted models.
        # Loading from config breaks `_check` because not all attributed needed for fitting are set.
        if not self.is_fitted:
            super()._check()
    
    def __eq__(self, other) -> bool:
        if not issubclass(type(other), MyHMM):
            return False
        if not self.is_fitted and not other.is_fitted:
            return self._compare_unfitted(other)
        elif self.is_fitted and other.is_fitted:
            return self._compare_fitted(other)
        return False

    def _compare_unfitted(self, other) -> bool:
        keys_to_compare = ['n_components', 'init_params', 'n_iter', 'tol']

        return all([
            self.init_config[k] == other.init_config[k] 
            for k in keys_to_compare
        ])
    
    def _compare_fitted(self, other) -> bool:
        return (
            np.array_equal(self.startprob_, other.startprob_) and
            np.array_equal(self.transmat_, other.transmat_) and
            np.array_equal(self.mapping, other.mapping)
        )

    def aic(
            self,
            X: np.ndarray,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray:
        return hmm.BaseHMM.aic(self, X, lengths)

    def bic(
            self,
            X: np.ndarray,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray:
        return hmm.BaseHMM.bic(self, X, lengths)
