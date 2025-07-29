import numpy as np
from . import Model


class Flat(Model):
    """
    Flat volatility model.
    """

    _name = "Flat"
    _param_names = [
        "sigma",
    ]
    _default_bounds = [(Model._epsilon, None)]  # sigma > 0

    def ivs(
        self,
        spot: float,
        k: np.ndarray,
        t: float,
        r: float,
        params: list[float] = None,
    ) -> np.ndarray:
        """
        Return model implied volatilities for a set of strikes.
        """
        if params is None:
            params = self.params
        else:
            params = self._parse_params(params)

        return np.full_like(k, params[0])  # Flat volatility
