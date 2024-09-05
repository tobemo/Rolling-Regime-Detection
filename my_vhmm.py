import json
import os

import numpy as np
from hmmlearn.vhmm import VariationalGaussianHMM

from base import MyHMM


class MyVariationalGaussianHMM(MyHMM, VariationalGaussianHMM):
    """VariationalGaussianHMM with some settings fixed:
    
    covariance_type: 'full'
    implementation: 'scaling'
    params: 'stmc'
    algorithm: 'viterbi'

    # s: start probability
    # t: transition matrix
    # m: mean
    # c: covariance matrix
    """
    @property
    def means_(self):
        return self.means_posterior_
    @means_.setter
    def means_(self, means) -> None:
        # monkey patch to make mean settable
        self.means_posterior_ = means
    
    def __init__(
            self,
            n_components=1,
            init_params='stmc',
            random_state=None,
            n_iter=100,
            tol=1e-6,
            verbose=False,
            timestamp: int = None,
        ) -> None:
        MyHMM.__init__(self, timestamp=timestamp)
        VariationalGaussianHMM.__init__(
            self,
            covariance_type="full",
            algorithm="viterbi",
            params='stmc',
            implementation='scaling',
            n_components=n_components, 
            init_params=init_params,
            random_state=random_state,
            n_iter=n_iter,
            tol=tol,
            verbose=verbose,
        )
    
    def get_config(self) -> dict:
        config = {
            "timestamp": self.timestamp,
            "n_components": self.n_components,
            "init_params": "" if self.is_fitted else self.init_params,
            "random_state": self.random_state,
            "n_iter": self.n_iter,
            "tol": self.tol,
            "verbose": self.verbose,
            "covariance_type": self.covariance_type,
            "startprob": self.startprob_.tolist(),
            "transmat": self.transmat_.tolist(),
            "means": self.means_.tolist(),
            "covars": self.covars_.tolist(),
            "n_features": self.n_features,
            "transition_cost": self.transition_cost,
        }
        if hasattr(self, 'mapping'):
            config["mapping"] = self.mapping.tolist()
        return config
    
    @classmethod
    def from_config(cls, config: dict):
        """Initialize a MyVariationalGaussianHMM from a config.
        
        See `to_config()`"""
        model = cls(
            n_components=config["n_components"],
            init_params=config["init_params"],
            random_state=config["random_state"],
            n_iter=config["n_iter"],
            tol=config["tol"],
            verbose=config["verbose"],
            timestamp=config["timestamp"],
        )
        model.n_features = config["n_features"]
        model.transition_cost = config["transition_cost"]
        if 'mapping' in config:
            model.mapping = np.array(config["mapping"])
        
        # assign parameters
        model.startprob_ = np.array(config["startprob"])
        model.transmat_ = np.array(config["transmat"])
        model.means_ = np.array(config["means"])
        model.covars_ = np.array(config["covars"])

        return model

    def __eq__(self, other) -> bool:
        if not super().__eq__(other):
            return False
        cond_a = np.array_equal(self.means_, other.means_)
        cond_b = np.array_equal(self.covars_, other.covars_)
        cond_c = self.n_features == other.n_features
        return cond_a and cond_b and cond_c

