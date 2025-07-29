import numpy as np
from . import Distribution


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
