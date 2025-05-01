import numpy as np
from typing import ClassVar
from pydantic import BaseModel, PrivateAttr, computed_field, model_validator
from scipy.optimize import basinhopping

from general.util import black76


class Model(BaseModel):
    """
    Model to represent a financial model given deterministic parameters.
    """

    _epsilon: ClassVar[float] = 1e-5

    _name: ClassVar[str] = PrivateAttr()

    @computed_field
    @property
    def name(self) -> str:
        if not self._name:
            raise ValueError("Child class must define _name.")
        return self._name

    _param_names: ClassVar[list[str]] = PrivateAttr()

    @computed_field
    @property
    def param_names(self) -> list[str]:
        if not self._param_names:
            raise ValueError("Child class must define _param_names.")
        return self._param_names

    _default_bounds: ClassVar[list[tuple]] = PrivateAttr()

    @computed_field
    @property
    def default_bounds(self) -> list[tuple]:
        if not self._default_bounds:
            raise ValueError("Child class must define _default_bounds.")
        return self._default_bounds

    params: list[float] = []
    _bounds: list[tuple] = PrivateAttr(default_factory=list)

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
    def _validate_params(self):
        self._bounds = self.default_bounds.copy()
        self._validate_params_with_bounds(self.params, self._bounds)
        return self

    def _validate_params_with_bounds(
        self, params: list[float], bounds: list[tuple]
    ) -> None:
        """
        Validate that the parameters are within the specified bounds.
        """
        if len(params) == 0:
            return
        if not bounds:
            bounds = self.default_bounds
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
        initial_parameters: list[float] | dict[str, float] | np.ndarray,
        fixed_params: dict[str, float] = {},
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

        self._validate_fixed_params(fixed_params)
        for name, value in fixed_params.items():
            idx = self.param_names.index(name)
            self.params[idx] = value
            self._bounds[idx] = (value, value)

        self._validate_params_with_bounds(self.params, self._bounds)

    def calibrate(
        self,
        spot: float,
        k: np.ndarray,
        t: float,
        r: float,
        market_ivs: np.ndarray,
        initial_parameters: list[float] | dict[str, float] | np.ndarray,
        fixed_params: dict[str, float] = {},
        n_iter: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Calibrates the model parameters using market data and saves them to params.
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

        self.set_params_and_bounds(initial_parameters, fixed_params)

        if verbose:
            print(f"[{self.name}] Initial parameters: {self.params}")
            print(f"[{self.name}] Bounds: {self._bounds}")

        result = basinhopping(
            objective,
            self.params,
            minimizer_kwargs={"method": "L-BFGS-B", "bounds": self._bounds},
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
        raise NotImplementedError(
            f"Method ivs not implemented for {self.__class__.__name__}. Please implement it in the subclass."
        )


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
        self, spot: float, K: np.ndarray, T: float, r: float, params: list[float] = None
    ) -> np.ndarray:
        """
        Return model implied volatilities for a set of strikes.
        """
        f = spot * np.exp(r * T)
        if params is None:
            params = self.params
        beta = self.params[self.param_names.index("beta")]
        alpha = self.params[self.param_names.index("alpha")]
        rho = self.params[self.param_names.index("rho")]
        gamma = self.params[self.param_names.index("gamma")]

        # We make sure that the input is of array type
        if type(K) is float:
            K = np.array([K])
        if type(K) is not np.ndarray:
            K = np.array(K).reshape([len(K), 1])
        z = gamma / alpha * np.power(f * K, (1.0 - beta) / 2.0) * np.log(f / K)
        x_z = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
        A = alpha / (
            np.power(f * K, ((1.0 - beta) / 2.0))
            * (
                1.0
                + np.power(1.0 - beta, 2.0) / 24.0 * np.power(np.log(f / K), 2.0)
                + np.power((1.0 - beta), 4.0) / 1920.0 * np.power(np.log(f / K), 4.0)
            )
        )
        B1 = (
            1.0
            + (
                np.power((1.0 - beta), 2.0)
                / 24.0
                * alpha
                * alpha
                / (np.power((f * K), 1 - beta))
                + 1
                / 4
                * (rho * beta * gamma * alpha)
                / (np.power((f * K), ((1.0 - beta) / 2.0)))
                + (2.0 - 3.0 * rho * rho) / 24.0 * gamma * gamma
            )
            * T
        )
        imp_vol = A * (z / x_z) * B1
        B2 = (
            1.0
            + (
                np.power(1.0 - beta, 2.0)
                / 24.0
                * alpha
                * alpha
                / (np.power(f, 2.0 - 2.0 * beta))
                + 1.0 / 4.0 * (rho * beta * gamma * alpha) / np.power(f, (1.0 - beta))
                + (2.0 - 3.0 * rho * rho) / 24.0 * gamma * gamma
            )
            * T
        )

        # Special treatment for ATM strike price
        if len(np.where(K == f)[0]) > 0:
            imp_vol[np.where(K == f)] = alpha / np.power(f, (1 - beta)) * B2
        return imp_vol.squeeze()
