import json
import os
from copy import deepcopy

import numpy as np
from hmmlearn.vhmm import VariationalGaussianHMM as VGHMM
from sklearn.utils.validation import check_is_fitted

from .base import HMMBase


class VariationalGaussianHMM(HMMBase, VGHMM):
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
    n_components: int
    @property
    def means_(self):
        return self.means_posterior_
    @means_.setter
    def means_(self, means) -> None:
        # monkey patch to make mean settable
        self.means_posterior_ = means

    def __init__(
            self,
            n_components=2,
            init_params='stmc',
            random_state=None,
            n_iter=100,
            tol=1e-6,
            verbose=False,
            name: str = 'Regime'
        ) -> None:
        """If random state is set, one model is fitted, if random state is None, k models are fitted and the best one is kept, see `fit` and `multi_fit`.`"""
        HMMBase.__init__(self, name=name)
        VGHMM.__init__(
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

    def get_fitted_params(self) -> dict:
        check_is_fitted(self, 'startprob_')
        params = super().get_fitted_params()
        params.update(
            {
                "init_params": "" if self.is_fitted else self.init_params, # !
                "covariance_type": self.covariance_type,
                "startprob_": self.startprob_.tolist(),
                "transmat_": self.transmat_.tolist(),
                "means_": self.means_.tolist(),
                "covars_": self.covars_.tolist(),
                "n_features": self.n_features,
            }
        )
        return params

    def __eq__(self, other) -> bool:
        if not super().__eq__(other):
            return False
        if self.is_fitted and other.is_fitted:
            return self._compare_fitted(other)
        if not self.is_fitted and not other.is_fitted:
            return super()._compare_unfitted(other)
        return False
    
    def _compare_fitted(self, other) -> bool:
        return (
            super()._compare_fitted(other) and
            np.array_equal(self.means_, other.means_) and
            np.array_equal(self.covars_, other.covars_) and
            self.n_features == other.n_features
        )

