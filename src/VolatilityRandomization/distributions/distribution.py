import numpy as np
from typing import ClassVar
from pydantic import BaseModel, PrivateAttr


class Distribution(BaseModel):
    """
    Base class for all distributions.
    """

    _epsilon: ClassVar[float] = 1e-5

    _name: ClassVar[str] = PrivateAttr()

    @property
    def name(self) -> str:
        if not self._name:
            raise ValueError("Child class must define _name.")
        return self._name

    _param_names: ClassVar[list[str]] = PrivateAttr()

    @property
    def param_names(self) -> list[str]:
        if not self._param_names:
            raise ValueError("Child class must define _param_names.")
        return self._param_names

    _bounds: ClassVar[list[tuple]] = PrivateAttr()

    @property
    def bounds(self) -> list[tuple]:
        if not self._bounds:
            raise ValueError("Child class must define _bounds.")
        return self._bounds

    def get_gram_matrix(self, n: int, params: list[float]) -> np.ndarray:
        """
        Returns the Gram matrix for the distribution.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def get_alphas(self, r):
        return [
            r[j, j + 1] / r[j, j] - r[j - 1, j] / r[j - 1, j - 1]
            for j in range(len(r) - 1)
        ]

    def get_betas(self, r):
        return [(r[j + 1, j + 1] / r[j, j]) ** 2 for j in range(0, len(r) - 2)]

    def collocation_points(self, n, params):
        """
        Returns the collocation points for the distribution.
        """
        m = self.get_gram_matrix(n, params)
        r = np.linalg.cholesky(m).T
        a = self.get_alphas(r)
        b = self.get_betas(r)
        j = np.diag(a) + np.diag(np.sqrt(b), 1) + np.diag(np.sqrt(b), -1)
        x, W = np.linalg.eig(j)
        w = W[0, :] ** 2
        idx = np.argsort(x)
        w = w[idx]
        x = x[idx]
        return w, x
