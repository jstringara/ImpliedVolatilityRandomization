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
        print(f"Gram matrix: {m.shape}")
        r = np.linalg.cholesky(m).T
        print(f"Cholesky factor: {r.shape}")
        a = self.get_alphas(r)
        print(f"Alphas: {a}")
        b = self.get_betas(r)
        j = np.diag(a) + np.diag(np.sqrt(b), 1) + np.diag(np.sqrt(b), -1)
        x, W = np.linalg.eig(j)
        w = W[0, :] ** 2
        idx = np.argsort(x)
        w = w[idx]
        x = x[idx]
        return w, x


class Gamma(Distribution):
    """
    Gamma distribution.
    """

    _name = "Gamma"
    _param_names = ["shape", "scale"]
    _bounds = [(Distribution._epsilon, None), (Distribution._epsilon, None)]

    def get_gram_matrix(self, n, params):
        """
        Returns the Gram matrix for the distribution.
        """
        k, theta = params
        m = np.zeros((n + 1, n + 1))

        for idx, _ in np.ndenumerate(m):
            ind = sum(idx)
            m[idx] = theta**ind * np.prod([k + i for i in range(ind)])

        return m


class Beta(Distribution):
    """
    Beta distribution.
    """

    _name = "Beta"
    _param_names = ["alpha", "beta"]
    _bounds = [(Distribution._epsilon, None), (Distribution._epsilon, None)]

    def get_gram_matrix(self, n, params):
        """
        Returns the Gram matrix for the distribution.
        """
        alpha, beta = params
        m = np.zeros((n + 1, n + 1))

        for idx, _ in np.ndenumerate(m):
            ind = sum(idx)
            m[idx] = np.prod([(alpha + i) / (alpha + beta + i) for i in range(ind)])

        return m


class Uniform(Distribution):
    """
    Uniform distribution.
    """

    _name = "Uniform"
    _param_names = ["lower", "upper"]
    _bounds = [(None, None), (None, None)]

    def get_gram_matrix(self, n, params):
        """
        Returns the Gram matrix for the distribution.
        """
        u, d = params
        return 1 / (
            np.linspace(np.ones(n + 1), np.ones(n + 1) * (n + 1), n + 1)
            + np.arange(n + 1)
        )


class Normal(Distribution):
    """
    Normal distribution.
    """

    _name = "Normal"
    _param_names = ["mean", "stddev"]
    _bounds = [(None, None), (Distribution._epsilon, None)]

    def get_gram_matrix(self, n, params):
        """
        Returns the Gram matrix for the distribution.
        """
        mu, eta = params
        m = np.zeros((n + 1, n + 1))

        for idx, _ in np.ndenumerate(m):
            ind = sum(idx)
            m[idx] = np.exp(ind * mu + 0.5 * ind**2 * eta)

        return m


class LogNormal(Distribution):
    """
    Log-normal distribution.
    """

    _name = "LogNormal"
    _param_names = ["mean", "stddev"]
    _bounds = [(None, None), (Distribution._epsilon, None)]

    def get_gram_matrix(self, n, params):
        """
        Returns the Gram matrix for the distribution.
        """
        mu, eta = params
        m = np.zeros((n + 1, n + 1))

        for idx, _ in np.ndenumerate(m):
            ind = sum(idx)
            m[idx] = np.exp(ind * mu + 0.5 * ind**2 * eta)

        return m
