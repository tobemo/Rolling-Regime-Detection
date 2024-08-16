import json
import os
import time

import numpy as np
from hmmlearn import vhmm

from base import MyHMM


class MyVariationalGaussianHMM(vhmm.VariationalGaussianHMM, MyHMM):
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
        self.timestamp = timestamp or int(time.time())
        super().__init__(
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

