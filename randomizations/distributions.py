import numpy as np
from pydantic import BaseModel, PrivateAttr


class Distribution(BaseModel):
    """
    Base class for all distributions.
    """

    _name: PrivateAttr[str]
    _parameter_names: PrivateAttr[list[str]]
    parameters: list[float]
    _bounds: PrivateAttr[list[tuple]]

    def get_gram_matrix(self):
        """
        Returns the Gram matrix for the distribution.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def get_alphas(r):
        return [r[j, j + 1] / r[j, j] - r[j - 1, j] / r[j - 1, j - 1] for j in range(len(r) - 1)]


    def get_betas(r):
        return [(r[j + 1, j + 1] / r[j, j]) ** 2 for j in range(0, len(r) - 2)]

    def collocation_points(self, n):
        """
        Returns the collocation points for the distribution.
        """
        m = self.get_gram_matrix(n)
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


class Gamma(Distribution):
    """
    Gamma distribution.
    """

    _name: str = "Gamma"
    _parameter_names: list[str] = ["shape", "scale"]
    parameters: list[float] = [1.0, 1.0]
    _bounds: list[tuple] = [(0, None), (0, None)]

    def get_gram_matrix(self, n):
        """
        Returns the Gram matrix for the distribution.
        """
        k, theta = self.parameters
        m = np.zeros((n + 1, n + 1))

        for idx, _ in np.ndenumerate(m):
            ind = sum(idx)
            m[idx] = theta**ind * np.prod([k + i for i in range(ind)])

        return m

class Beta(Distribution):
    """
    Beta distribution.
    """

    _name: str = "Beta"
    _parameter_names: list[str] = ["alpha", "beta"]
    parameters: list[float] = [1.0, 1.0]
    _bounds: list[tuple] = [(0, None), (0, None)]

    def get_gram_matrix(self, n):
        """
        Returns the Gram matrix for the distribution.
        """
        alpha, beta = self.parameters
        m = np.zeros((n + 1, n + 1))

        for idx, _ in np.ndenumerate(m):
            ind = sum(idx)
            m[idx] = np.prod([(alpha + i) / (alpha + beta + i) for i in range(ind)])

        return m

class Uniform(Distribution):
    """
    Uniform distribution.
    """

    _name: str = "Uniform"
    _parameter_names: list[str] = ["lower", "upper"]
    parameters: list[float] = [0.0, 1.0]
    _bounds: list[tuple] = [(None, None), (None, None)]

    def get_gram_matrix(self, n):
        """
        Returns the Gram matrix for the distribution.
        """
        u, d = self.parameters
        return 1 / (np.linspace(np.ones(n + 1), np.ones(n + 1) * (n + 1), n + 1) + np.arange(n + 1))
        
        
class Normal(Distribution):
    """
    Normal distribution.
    """

    _name: str = "Normal"
    _parameter_names: list[str] = ["mean", "stddev"]
    parameters: list[float] = [0.0, 1.0]
    _bounds: list[tuple] = [(None, None), (0, None)]

    def get_gram_matrix(self, n):
        """
        Returns the Gram matrix for the distribution.
        """
        mu, eta = self.parameters
        m = np.zeros((n + 1, n + 1))

        for idx, _ in np.ndenumerate(m):
            ind = sum(idx)
            m[idx] = np.exp(ind * mu + 0.5 * ind**2 * eta)

        return m

class LogNormal(Distribution):
    """
    Log-normal distribution.
    """

    _name: str = "LogNormal"
    _parameter_names: list[str] = ["mean", "stddev"]
    parameters: list[float] = [0.0, 1.0]
    _bounds: list[tuple] = [(None, None), (0, None)]

    def get_gram_matrix(self, n):
        """
        Returns the Gram matrix for the distribution.
        """
        mu, eta = self.parameters
        m = np.zeros((n + 1, n + 1))

        for idx, _ in np.ndenumerate(m):
            ind = sum(idx)
            m[idx] = np.exp(ind * mu + 0.5 * ind**2 * eta)

        return m
    