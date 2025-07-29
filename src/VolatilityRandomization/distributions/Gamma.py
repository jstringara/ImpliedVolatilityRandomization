import numpy as np
from . import Distribution


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
