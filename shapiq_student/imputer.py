import numpy as np
from shapiq.games.imputer.base import Imputer

class GaussianImputer(Imputer):

    def __init__(self, model, data: np.ndarray, x: np.ndarray | None = None):
        self.model = model
        self.data = data
        self._x = x
        self.n_players = data.shape[1]
        self.randomState = np.random.RandomState(42)
        self.verbose = False

    def fit(self, x: np.ndarray):
        self._x = x
        return self

    # Using imputer object as a method
    def __call__(self, coalitions: np.ndarray, verbose: bool = False) -> np.ndarray:
        if isinstance(coalitions, list):
            coalitions = np.array(coalitions)
        if coalitions.dtype != bool:
            coalitions = coalitions.astype(bool)

        n_coalitions, n_features = coalitions.shape
        predictions = np.zeros(n_coalitions)

        for i, coalition in enumerate(coalitions):
            x_masked = self._x.copy()
            x_masked[0, ~coalition] = 0  # set inactive features to 0
            predictions[i] = self.model(x_masked)[0]

        return predictions


class GaussianCopulaImputer(Imputer):

    def __init__(self, model, data: np.ndarray, x: np.ndarray | None = None):
        self.model = model
        self.data = data
        self._x = x
        self.n_players = data.shape[1]
        self.randomState = np.random.RandomState(42)
        self.verbose = False

    def fit(self, x: np.ndarray):
        self._x = x
        return self

    # Using imputer object as a method
    def __call__(self, coalitions: np.ndarray, verbose: bool = False) -> np.ndarray:
        if isinstance(coalitions, list):
            coalitions = np.array(coalitions)
        if coalitions.dtype != bool:
            coalitions = coalitions.astype(bool)

        n_coalitions, n_features = coalitions.shape
        predictions = np.zeros(n_coalitions)

        for i, coalition in enumerate(coalitions):
            x_masked = self._x.copy()
            x_masked[0, ~coalition] = 0  # set inactive features to 0
            predictions[i] = self.model(x_masked)[0]

        return predictions


