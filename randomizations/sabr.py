import numpy as np
from typing import ClassVar, Optional, Union
from pydantic import BaseModel, Field, model_validator
from scipy.optimize import basinhopping

from general.hagan import hagan_implied_volatility
from general.util import black76


class SABR(BaseModel):
    """
    Classical deterministic SABR model.
    """

    # Parameter names
    param_names: ClassVar[list[str]] = ["beta", "alpha", "rho", "gamma"]

    # Bounds for the parameters
    default_bounds: ClassVar[list[tuple[float, Optional[float]]]] = [
        (0.0, 1.0),  # beta in [0,1]
        (1e-6, None),  # alpha > 0
        (-1 + 1e-6, 1 - 1e-6),  # rho in (-1,1)
        (1e-6, None),  # gamma > 0
    ]

    # Parameters
    params: Optional[list[float]] = Field(default_factory=list)
    fixed_params: dict[str, float] = Field(default_factory=dict)
    bounds: list[tuple[float, Optional[float]]] = Field(default_factory=list)

    @classmethod
    def params_dict_to_list(cls, params_dict) -> list[float]:
        """
        Convert the parameters dictionary to a list.
        """
        if len(params_dict) != len(cls.param_names):
            raise ValueError(
                f"Expected {len(cls.param_names)} parameters, got {len(params_dict)}"
            )
        if params_dict.keys() != set(cls.param_names):
            raise ValueError(
                f"Expected parameters {cls.param_names}, got {params_dict.keys()}"
            )
        return [params_dict[name] for name in cls.param_names]

    @model_validator(mode="before")
    @classmethod
    def _parse_input_params(cls, values):
        raw_params = values.get("params")
        if isinstance(raw_params, dict):
            values["params"] = cls.params_dict_to_list(raw_params)
        elif raw_params is None:
            values["params"] = []
        return values

    @model_validator(mode="after")
    def _validate_params(self) -> "SABR":
        self._validate_params_with_bounds(self.params, self.bounds)
        self._validate_fixed_params(self.fixed_params)
        return self

    def _validate_params_with_bounds(
        self, params: list[float], bounds: list[tuple[float, Optional[float]]]
    ) -> None:
        """
        Validate that the parameters are within the specified bounds.
        """
        if len(params) == 0:
            return
        for i, (param, bound) in enumerate(zip(params, bounds)):
            if (bound[0] is not None and param < bound[0]) or (
                bound[1] is not None and param > bound[1]
            ):
                raise ValueError(
                    f"Parameter {self.param_names[i]} out of bounds: {param} not in {bound}"
                )

    def _validate_fixed_params(self, fixed_params: dict[str, float]) -> None:
        """
        Validate that the fixed parameters are valid.
        """
        for key in fixed_params:
            if key not in self.param_names:
                raise ValueError(
                    f"Invalid fixed parameter name: {key}. Expected one of {self.param_names}."
                )

    def to_dict(self) -> dict[str, float]:
        """
        Return parameters as a dictionary.
        """
        if len(self.params) == 0:
            raise ValueError("Parameters not set.")
        return {name: self.params[i] for i, name in enumerate(self.param_names)}

    def set_params_and_bounds(
        self,
        initial_parameters: Union[list[float], dict[str, float]],
    ) -> None:
        """
        Set the parameters and bounds, fixing any parameters as needed.
        """
        if isinstance(initial_parameters, dict):
            initial_parameters = self.params_dict_to_list(initial_parameters)
        if len(initial_parameters) != len(self.param_names):
            raise ValueError(
                f"Expected {len(self.param_names)} parameters, got {len(initial_parameters)}"
            )

        self.params = initial_parameters.copy()
        self.bounds = self.default_bounds.copy()

        for name, value in self.fixed_params.items():
            idx = self.param_names.index(name)
            self.params[idx] = value
            self.bounds[idx] = (value, value)

        self._validate_params_with_bounds(self.params, self.bounds)
        self._validate_fixed_params(self.fixed_params)

    def calibrate(
        self,
        spot: float,
        k: np.ndarray,
        t: float,
        r: float,
        market_ivs: np.ndarray,
        initial_parameters: Union[list[float], dict[str, float]],
        n_iter: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Calibrates the SABR model parameters using market data and saves them to params.
        spot: Current spot price
        k: Strike prices
        t: Time to maturity
        r: Risk-free interest rate
        market_ivs: Market implied volatilities
        initial_parameters: Initial guess for the model parameters
        """

        def objective(params: list) -> float:
            self.params = params
            try:
                model_ivs = np.array(self.ivs(spot, k, t, r))
                return np.mean((model_ivs - market_ivs) ** 2)
            except Exception as e:
                print(f"[SABR] Error in objective function: {e} with params: {params}")
                return np.inf

        self.set_params_and_bounds(initial_parameters)

        if verbose:
            print(f"[SABR] Initial parameters: {self.params}")
            print(f"[SABR] Bounds: {self.bounds}")

        result = basinhopping(
            objective,
            self.params,
            minimizer_kwargs={"method": "L-BFGS-B", "bounds": self.bounds},
            niter=n_iter,
            disp=verbose,
        )
        self.params = result.x.tolist()

    def prices(self, spot: float, k: np.ndarray, t: float, r: float) -> np.ndarray:
        """
        Return model prices for a set of strikes.
        """
        f = spot * np.exp(r * t)
        ivs = self.ivs(spot, k, t, r)
        return black76(f, k, t, r, ivs)

    def ivs(self, spot: float, k: np.ndarray, t: float, r: float) -> np.ndarray:
        """
        Return model implied volatilities for a set of strikes.
        """
        f = spot * np.exp(r * t)
        return hagan_implied_volatility(k, t, f, *self.params)
