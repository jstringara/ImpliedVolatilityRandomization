import numpy as np
from . import Distribution


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
