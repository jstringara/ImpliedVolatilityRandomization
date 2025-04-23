import numpy as np
from pydantic import BaseModel
from scipy.optimize import basinhopping

from general.collocation import get_col_points
from general.hagan import hagan_implied_volatility
from general.expansions import get_sigma_0, get_sigma_approx
from general.util import black76

nCol = 2


class RandSABR(BaseModel):
    # Randomizes the gamma parameter (vol of vol) with a Gamma random variable of two parameters
    params_rand: list = None
    n_col_points: int = nCol

    # Bounds for the parameters
    params_bounds: list = [
        (1e-6, None),   # alpha > 0
        (0.0, 1.0),     # beta in [0,1]
        (-1 + 1e-6, 1 - 1e-6),  # rho in (-1,1)
        (1e-6, None),   # gamma > 0
        (1e-6, None)    # scale > 0
    ]

    def calibrate(
        self,
        spot: float,
        k: np.ndarray,
        t: float,
        r: float,
        market_ivs: np.ndarray,
        initial_parameters: list[float],
        n_iter: int = 10
    ) -> None:
        """
        Calibrates the RandSABR model parameters using market data and saves them to params_rand.
        spot: Current spot price
        k: Strike prices
        t: Time to maturity
        r: Risk-free interest rate
        market_ivs: Market implied volatilities
        initial_parameters: Initial guess for the model parameters
        """
        def objective(params: list) -> float:
            self.params_rand = params
            try:
                model_ivs = np.array(self.ivs(spot, k, t, r))
                return np.mean((np.array(model_ivs) - market_ivs) ** 2)
            except Exception as e:
                print(f"[RandSABR] Error in objective function: {e}")
                return np.inf

        # Parameter bounds
        result = basinhopping(
            objective,
            initial_parameters,
            minimizer_kwargs={"method": "L-BFGS-B",
                              "bounds": self.params_bounds},
            niter=n_iter
        )
        self.params_rand = result.x

    def prices(self, spot: float, k: np.ndarray, t: float, r: float) -> np.ndarray:
        return self._mixed_hagan_prices(spot, k, t, r)

    def ivs(self, spot: float, k: np.ndarray, t: float, r: float) -> np.ndarray:
        return self._mixed_hagan_ivs(spot, k, t, r)

    def _generate_param_sets(self) -> tuple[np.ndarray, list[list[float]]]:
        non_randomized, gamma_params = self.params_rand[:3], self.params_rand[-2:]
        weights, col_points = get_col_points(self.n_col_points, gamma_params, "gamma")
        param_sets = [(*non_randomized, c) for c in col_points]
        return weights, param_sets

    def _mixed_hagan_ivs(
        self, spot: float, k: np.ndarray, t: float, r: float
    ) -> list[float]:
        weights, params = self._generate_param_sets()

        modelIVForParams = [np.atleast_1d(hagan_implied_volatility(
            k, t, spot * np.exp(r * t), *p)) for p in params]
        ivsMixed = []
        for strikeN, strike in enumerate(k):
            m = np.log(spot / strike) + r * t
            ivsAtStrike = np.array([ivs[strikeN] for ivs in modelIVForParams])
            sigma0 = get_sigma_0(t, weights, ivsAtStrike)
            ivsMixed.append(get_sigma_approx(
                m, t, sigma0, weights, ivsAtStrike))
        return ivsMixed

    def _mixed_hagan_prices(
        self, spot: float, k: np.ndarray, t: float, r: float
    ) -> np.ndarray:
        weights, params = self._generate_param_sets()
        
        f = spot * np.exp(r * t)
        pricesForParams = [
            black76(f, k, t, r, hagan_implied_volatility(k, t, f, *p)) for p in params]
        return (weights[:, np.newaxis] * pricesForParams).sum(axis=0)
