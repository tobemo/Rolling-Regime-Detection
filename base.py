import json
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


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
    """Creation time."""

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
                self.mapping.T.tolist()
            )
        )
    
    def __init__(self, timestamp: int = None) -> None:
        self.timestamp = timestamp or int(time.time())
        self._transition_cost = np.inf
        # mapping defaults to mapping to itself
    
    def predict(
            self,
            X: np.ndarray,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray:
        """Find most likely state sequence corresponding to ``X``. States are mapped using self.mapper."""
        predictions = super().predict(X, lengths=lengths)
        for key, value in self.mapper.items():
            predictions[predictions == key] = -1 * value
        predictions = predictions.abs()
        return predictions
    
    def predict_proba(
            self,
            X: np.ndarray,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray:
        probas = super().predict_proba(X, lengths=lengths)
        reordering = self.mapping[:,1]
        return probas[:, reordering]

    @property
    def config(self) -> dict:
        """Returns creation config."""
        return {
            "n_components": self.n_components,
            "init_params": self.init_params,
            "random_state": self.random_state,
            "n_iter": self.random_state,
            "tol": self.tol,
            "verbose": self.verbose,
            "timestamp": self.timestamp
        }

    @abstractmethod
    def get_config(self) -> dict:
        """Get config that exactly describes this model.
        This not only includes thins like creation time and number of components but also things like start probabilities.
        
        Can be used to initialize this class to an already fitted, and ready to use, model; when using MyVariationalGaussianHMM.from_config()
        
        This enables serialization and persistence."""

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

