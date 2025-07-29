import numpy as np
from . import Distribution


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
