import numpy as np
from . import Model
from VolatilityRandomization.general.hagan import hagan_implied_volatility


class SABR(Model):
    """
    Classical deterministic SABR model.
    """

    _name = "SABR"
    _param_names = ["beta", "alpha", "rho", "gamma"]
    _default_bounds = [
        (0.0, 1.0),  # beta in [0,1]
        (Model._epsilon, None),  # alpha > 0
        (-1 + Model._epsilon, 1 - Model._epsilon),  # rho in (-1,1)
        (Model._epsilon, None),  # gamma > 0
    ]

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
        f = spot * np.exp(r * t)
        if params is None:
            params = self.params
        else:
            params = self._parse_params(params)

        return hagan_implied_volatility(k, t, f, *params)
