import json
import time
from abc import ABC, abstractmethod

import numpy as np


class MyHMM:
    @property
    def is_fitted(self) -> bool:
        return hasattr(self, 'transmat_')
    
    _transition_cost = np.inf
    @property
    def transition_cost(self) -> float:
        """Transition cost from a unspecified model to this one. Assumes HMMs are use in a chain so the transition cost from the previous model to this one is known."""
        return  self._transition_cost
    @transition_cost.setter
    def transition_cost(self, cost: float) -> None:
        self._transition_cost = cost

    timestamp: int
    """Creation time."""
    
    def __init__(self, timestamp: int = None) -> None:
        self.timestamp = timestamp or int(time.time())

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

