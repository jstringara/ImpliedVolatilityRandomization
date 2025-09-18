import numpy as np
from typing import ClassVar
from pydantic import BaseModel, PrivateAttr, computed_field, model_validator
from scipy.optimize import basinhopping

from VolatilityRandomization.general.util import black76


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
    def _parse_params(
        cls, params: np.ndarray | tuple | dict | list | None
    ) -> list[float]:
        """
        Parse and convert the input parameters into a list of floats.
        Supports numpy arrays, tuples, dictionaries, lists, or None.
        """
        if params is None:
            return []

        if isinstance(params, np.ndarray):
            if len(params) != len(cls._param_names):
                raise ValueError(
                    f"Expected {len(cls._param_names)} parameters, got {len(params)}"
                )
            return params.tolist()

        if isinstance(params, tuple):
            if len(params) != len(cls._param_names):
                raise ValueError(
                    f"Expected {len(cls._param_names)} parameters, got {len(params)}"
                )
            return list(params)

        if isinstance(params, dict):
            if len(params) != len(cls._param_names):
                raise ValueError(
                    f"Expected {len(cls._param_names)} parameters, got {len(params)}"
                )
            if params.keys() != set(cls._param_names):
                raise ValueError(
                    f"Expected parameters {cls._param_names}, got {params.keys()}"
                )
            return [params[name] for name in cls._param_names]

        if isinstance(params, list):
            if len(params) != len(cls._param_names):
                raise ValueError(
                    f"Expected {len(cls._param_names)} parameters, got {len(params)}"
                )
            return params

        raise ValueError(
            f"Invalid type for params: {type(params)}. Expected list, tuple, dict, numpy array, or None."
        )

    @model_validator(mode="before")
    @classmethod
    def _parse_input_params(cls, values):
        values["params"] = cls._parse_params(values.get("params"))
        return values

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

    @model_validator(mode="after")
    def _validate_params(self):
        self._bounds = self.default_bounds.copy()
        self._validate_params_with_bounds(self.params, self._bounds)
        return self

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
        initial_parameters = self._parse_params(initial_parameters)

        self.params = initial_parameters.copy()
        self._bounds = self.default_bounds.copy()

        self._validate_fixed_params(fixed_params)
        for name, value in fixed_params.items():
            idx = self.param_names.index(name)
            self.params[idx] = value
            self._bounds[idx] = (value, value)

        self._validate_params_with_bounds(self.params, self._bounds)

    def prices(
        self,
        spot: float,
        k: np.ndarray,
        t: float,
        r: float,
        params: list | dict | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return model prices for a set of strikes.
        """
        if params is None:
            params = self.params
        else:
            params = self._parse_params(params)

        f = spot * np.exp(r * t)
        ivs = self.ivs(spot, k, t, r, params)
        return black76(f, k, t, r, ivs)

    def ivs(
        self,
        spot: float,
        k: np.ndarray,
        t: float,
        r: float,
        params: list | dict | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return model implied volatilities for a set of strikes.
        """
        raise NotImplementedError(
            f"Method ivs not implemented for {self.__class__.__name__}. Please implement it in the subclass."
        )

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
    ) -> float:
        """
        Calibrates the model parameters using market data and saves them to params.
        spot: Current spot price
        k: Strike prices
        t: Time to maturity
        r: Risk-free interest rate
        market_ivs: Market implied volatilities
        initial_parameters: Initial guess for the model parameters
        """

        self.set_params_and_bounds(initial_parameters, fixed_params)

        if verbose:
            print(f"[{self.name}] Initial parameters: {self.params}")
            print(f"[{self.name}] Bounds: {self._bounds}")

        def objective(params: list) -> float:
            try:
                model_ivs = np.array(self.ivs(spot, k, t, r, params))
                return np.mean((model_ivs - market_ivs) ** 2)
            except Exception as e:
                print(
                    f"[{self.name}] Error in objective function: {e} with params: {params}"
                )
                return np.inf

        result = basinhopping(
            objective,
            self.params,
            minimizer_kwargs={"method": "L-BFGS-B", "bounds": self._bounds},
            niter=n_iter,
            disp=verbose,
        )
        self.params = result.x.tolist()

        return result.fun
