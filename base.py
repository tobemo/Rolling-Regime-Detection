import json
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn import hmm
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


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
            X: np.ndarray | pd.DataFrame,
            lengths: Optional[list[int]] = None,
            k: int = 5,
        ):
        score = -np.inf
        model = None
        config = self.HMM_config
        for i in range(1, k+1):
            config['random_state'] = np.random.randint(1e6)
            _model = self.HMM(**config)
            try:
                _model.fit(X, lengths)
            except Exception as e:
                # raise on last fit only, and only if no model has been fitted
                if not model and i == k:
                    raise e
                continue
            _score = _model.score(X)
            if _score > score:
                model = _model
                score = _score
        
        if not model:
            raise RuntimeError("Failed to fit a single model.")
        self.__dict__.update(model.__dict__)
        return self
        
    def fit(
            self,
            X: np.ndarray | pd.DataFrame,
            lengths: Optional[list[int]] = None,
            k: int = 5,
        ):
        """Estimate model parameters."""
        if self.random_state:
            super().fit(X, lengths)
        else:
            self._multi_fit(X, lengths, k)
    
        self.timestamp = int(time.time())
        if isinstance(X, pd.DataFrame) and isinstance(X.index,pd.DatetimeIndex):
            self.timestamp = int(X.index[-1].timestamp())

        return self
    
    def map_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Map predictions using `self.mapping`.
        Values in predictions that equal the second column of mapping are replaced by the respective value in the first column of mapping."""
        return np.select(
            [predictions == i for i in self.mapping[:,1]],
            self.mapping[:,0]
        )
    
    def _predict(
            self,
            X: np.ndarray | pd.DataFrame,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray | pd.Series:
        """Find most likely state sequence corresponding to ``X``.
        This call is without mapping, under the hood it just calls the original predict method.."""
        assert self.is_fitted, "Model is not fitted."
        predictions = super().predict(X, lengths=lengths)
        if isinstance(X, pd.DataFrame):
            predictions = pd.Series(predictions, index=X.index, name='Regime')
        return predictions

    def predict(
            self,
            X: np.ndarray | pd.DataFrame,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray | pd.Series:
        """Find most likely state sequence corresponding to ``X``.
        States are mapped using self.mapper."""
        predictions = self._predict(X, lengths)
        if isinstance(predictions, pd.Series):
            predictions = predictions.to_numpy()
        predictions = self.map_predictions(predictions)
        if isinstance(X, pd.DataFrame):
            predictions = pd.Series(predictions, index=X.index, name='Regime')
        return predictions
    
    def fit_predict(
            self,
            X: np.ndarray | pd.DataFrame,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray | pd.Series:
        """Estimate model parameters, then find most likely state sequence."""
        self.fit(X, lengths=lengths)
        return self.predict(X, lengths=lengths)
    
    def predict_proba(
            self,
            X: np.ndarray | pd.DataFrame,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray | pd.DataFrame:
        """Compute the posterior probability for each state in the model."""
        if hasattr(self, '_mapping'):
            raise NotImplementedError(
                "Using predict_proba while having a mapper set is not supported."
            )
        assert self.is_fitted, "Model is not fitted."
        probas = super().predict_proba(X, lengths=lengths)
        reordering = self.mapping[:,1]
        probas = probas[:, reordering]
        if isinstance(X, pd.DataFrame):
            probas = pd.DataFrame(probas, index=X.index)
        return probas

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
            X: np.ndarray | pd.DataFrame,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray:
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return hmm.BaseHMM.aic(self, X, lengths)

    def bic(
            self,
            X: np.ndarray | pd.DataFrame,
            lengths: Optional[list[int]]=None
        ) -> np.ndarray:
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return hmm.BaseHMM.bic(self, X, lengths)

    @staticmethod
    def _scatter_1D(
        X: np.ndarray | pd.DataFrame,
        Z: np.ndarray | pd.DataFrame
        ) -> plt.Axes:
        """Plots regime data over time plus a violin plot."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            Z = pd.DataFrame(Z)

        fig, ax = plt.subplots(
            ncols=2,
            sharey=True,
            width_ratios=[0.8, 0.2],
            gridspec_kw=dict(wspace=0)
        )
        ax[0].scatter(X.index, X, c=Z, cmap='rocket')

        parts = ax[1].violinplot(X, showextrema=False, showmedians=False)
        for pc in parts['bodies']:
            pc.set_facecolor('grey')
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        ax[1].scatter([1]*len(X), X, c=Z, cmap='rocket')
        ax[1].tick_params(
            axis='x', which='both',
            bottom=False, labelbottom=False,
        )
        return ax
    
    def scatter_1D(
        self,
        X: np.ndarray | pd.DataFrame,
        lengths: Optional[list[int]]=None
    ) -> plt.Axes:
        """Emits regimes for X then plots regime data over time plus a violin plot."""
        Z = self.predict(X)
        return self._scatter_1D(X=X, Z=Z)
    
    @staticmethod
    def _scatter_2D(
        X: np.ndarray | pd.DataFrame,
        Z: np.ndarray | pd.DataFrame
        ) -> plt.Axes:
        """Plots a scatter plot plus a histogram."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            Z = pd.DataFrame(Z)
        
        fig, ax = plt.subplots(
            ncols=2,
            width_ratios=[0.8, 0.2],
            gridspec_kw=dict(wspace=0.05)
        )
        scatter = ax[0].scatter(X.iloc[:,0], X.iloc[:, 1], c=Z, cmap='rocket')

        counts = Z.value_counts().sort_index()
        counts.plot.bar(
            ax=ax[1],
            color=[scatter.to_rgba(i) for i in counts.index],
        )
        ax[1].yaxis.set_label_position("right")
        ax[1].yaxis.tick_right()
        return ax

    def scatter_2D(
        self,
        X: np.ndarray | pd.DataFrame,
        lengths: Optional[list[int]]=None
    ) -> plt.Axes:
        """Emits regimes for X then plots a scatter plot plus a histogram."""
        Z = self.predict(X)
        return self._scatter_2D(X=X, Z=Z)

    def scatter_nD(
        self,
        X: np.ndarray | pd.DataFrame,
        lengths: Optional[list[int]]=None
        ) -> plt.Axes:
        """Emits regimes for X then reduces X, using t-SNE, and plots a scatter plot plus a histogram."""
        Z = self.predict(X)
        X_embedded = TSNE(self.n_components).fit_transform(X)
        return self._scatter_2D(X=X_embedded, Z=Z)        

    def scatter(
        self,
        X: np.ndarray | pd.DataFrame,
        lengths: Optional[list[int]]=None
        ) -> plt.Axes:
        """Plot regime data."""
        if X.shape[1] == 1:
            return self.scatter_1D(X, lengths=lengths)
        elif X.shape[1] == 2:
            return self.scatter_2D(X, lengths=lengths)
        else:
            return self.scatter_nD(X, lengths=lengths)
