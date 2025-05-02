import numpy as np
from typing import ClassVar, List, Dict, Optional
from pydantic import BaseModel, model_validator, PrivateAttr, computed_field
from scipy.optimize import basinhopping

from general.expansions import get_sigma_0, get_sigma_approx

from randomizations.distributions import Distribution
from randomizations.models import Model


class RandomizedModel(BaseModel):
    """
    Randomized model for option pricing and implied volatility.
    """

    # Class variables
    n_col_points: ClassVar[int] = 2

    model: Model
    distribution: Distribution
    randomized_param: str

    # Attributes to be computed
    _name: str = PrivateAttr(default_factory=str)

    @computed_field
    @property
    def name(self) -> str:
        """
        Return the name of the model.
        """
        if not self._name:
            self._name = (
                f"{self.model.name}_{self.distribution.name}_{self.randomized_param}"
            )
        return self._name

    _param_names: list[str] = PrivateAttr(default_factory=list)

    @computed_field
    @property
    def param_names(self) -> List[str]:
        """
        Return the names of the parameters.
        """
        if not self._param_names:
            param_names, start_randomized_params = self.compute_param_names(
                self.distribution, self.model, self.randomized_param
            )
            self._param_names = param_names
            self._start_randomized_param = start_randomized_params
        return self._param_names

    _start_randomized_param: int = PrivateAttr(default_factory=int)

    @computed_field
    @property
    def start_randomized_param(self) -> int:
        """
        Return the index of the randomized parameter.
        """
        if not self._start_randomized_param:
            self._start_randomized_param = len(self.model.param_names) - 1

        return self._start_randomized_param

    @property
    def randomized_params(self) -> List[float]:
        """
        Returns the parameters of the randomized model's distribution.
        """
        return self.params[self.start_randomized_param :]

    @property
    def non_randomized_params(self) -> List[float]:
        """
        Returns the non-randomized parameters of the model.
        """
        return self.params[: self.start_randomized_param]

    _default_bounds: List[tuple] = PrivateAttr(default_factory=list)

    @computed_field
    @property
    def default_bounds(self) -> List[tuple]:
        """
        Return the default bounds for the parameters.
        """
        if not self._default_bounds:
            # get the default bounds from the model
            self._default_bounds = self.model.default_bounds.copy()
            # find the position of the randomized parameter
            idx = self.model.param_names.index(self.randomized_param)
            # remove the randomized parameter from the list
            self._default_bounds.pop(idx)
            # add the distribution parameters at the end
            self._default_bounds += self.distribution.bounds.copy()
        return self._default_bounds

    # Parameters
    params: Optional[list[float]] = []
    _bounds: List[tuple] = PrivateAttr(default_factory=list)

    @staticmethod
    def _parse_params(
        params: np.ndarray | tuple | dict | list | None, param_names: list[str]
    ) -> List[float]:
        """
        Parse and convert the input parameters into a list of floats.
        Supports numpy arrays, tuples, dictionaries, lists, or None.
        """
        if params is None:
            return []

        if isinstance(params, np.ndarray):
            if len(params) != len(param_names):
                raise ValueError(
                    f"Expected {len(param_names)} parameters, got {len(params)}"
                )
            return params.tolist()

        if isinstance(params, tuple):
            if len(params) != len(param_names):
                raise ValueError(
                    f"Expected {len(param_names)} parameters, got {len(params)}"
                )
            return list(params)

        if isinstance(params, dict):
            if len(params) != len(param_names):
                raise ValueError(
                    f"Expected {len(param_names)} parameters, got {len(params)}"
                )
            if params.keys() != set(param_names):
                raise ValueError(
                    f"Expected parameters {param_names}, got {params.keys()}"
                )
            return [params[name] for name in param_names]

        if isinstance(params, list):
            if len(params) != len(param_names):
                raise ValueError(
                    f"Expected {len(param_names)} parameters, got {len(params)}"
                )
            return params

        raise ValueError(
            f"Invalid type for params: {type(params)}. Expected list, tuple, dict, numpy array, or None."
        )

    @staticmethod
    def compute_param_names(
        distribution: Distribution, model: Model, randomized_param: str
    ) -> List[str]:
        """
        Compute the parameter names for the randomized model.
        """
        param_names = model.param_names.copy()
        if randomized_param not in param_names:
            raise ValueError(
                f"Randomized parameter {randomized_param} not found in model parameters."
            )
        # remove the randomized parameter from the list
        param_names.remove(randomized_param)
        start_randomized_param = len(param_names)
        # add the distribution parameters at the end
        param_names += distribution.param_names.copy()
        return param_names, start_randomized_param

    @model_validator(mode="before")
    @classmethod
    def _parse_input(cls, values):
        param_names, _ = cls.compute_param_names(
            values["distribution"], values["model"], values["randomized_param"]
        )
        values["params"] = cls._parse_params(values.get("params"), param_names)
        return values

    def _validate_params_with_bounds(
        self, params: List[float], bounds: List[tuple]
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

    def _validate_fixed_params(self, fixed_params: Dict[str, float]) -> None:
        """
        Validate that the fixed parameters are valid.
        """
        for key in fixed_params:
            if key not in self.param_names:
                raise ValueError(
                    f"Invalid fixed parameter name: {key}. Expected one of {self.param_names}."
                )

    @model_validator(mode="after")
    def _validate_params(self):
        self._bounds = self.default_bounds.copy()
        self._validate_params_with_bounds(self.params, self._bounds)

        return self

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
        self._bounds = self.default_bounds.copy()

        self._validate_fixed_params(fixed_params)
        for name, value in fixed_params.items():
            idx = self.param_names.index(name)
            self.params[idx] = value
            self._bounds[idx] = (value, value)

        self._validate_params_with_bounds(self.params, self._bounds)

    def _generate_param_sets(self) -> tuple[np.ndarray, list[list[float]]]:
        """
        Generate parameter sets by collocating on the distribution.
        """
        weights, col_points = self.distribution.collocation_points(
            self.n_col_points, self.randomized_params
        )
        param_sets = [(*self.non_randomized_params, c) for c in col_points]
        return weights, param_sets

    def prices(self, spot: float, k: np.ndarray, t: float, r: float) -> np.ndarray:
        """
        Return model prices for a set of strikes.
        Compute prices by mixing over randomization.
        """
        weights, params = self._generate_param_sets()
        prices_for_params = [self.model.prices(spot, k, t, r, p) for p in params]
        return (weights[:, np.newaxis] * prices_for_params).sum(axis=0)

    def ivs(self, spot: float, k: np.ndarray, t: float, r: float) -> np.ndarray:
        """
        Return model implied volatilities for a set of strikes.
        Compute implied volatilities by mixing over randomization.
        """
        weights, params = self._generate_param_sets()
        model_ivs = [np.atleast_1d(self.model.ivs(spot, k, t, r, p)) for p in params]

        ivs_mixed = []
        for strike_idx, strike in enumerate(k):
            m = np.log(spot / strike) + r * t
            ivs_at_strike = np.array([ivs[strike_idx] for ivs in model_ivs])
            sigma0 = get_sigma_0(t, weights, ivs_at_strike)
            ivs_mixed.append(get_sigma_approx(m, t, sigma0, weights, ivs_at_strike))
        return ivs_mixed

    def calibrate(
        self,
        spot: float,
        k: np.ndarray,
        t: float,
        r: float,
        market_ivs: np.ndarray,
        initial_parameters: list | np.ndarray | tuple | dict,
        fixed_params: dict[str, float] = {},
        n_iter: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Calibrates the RandSABR model parameters using market data and saves them to params.
        spot: Current spot price
        k: Strike prices
        t: Time to maturity
        r: Risk-free interest rate
        market_ivs: Market implied volatilities
        initial_parameters: Initial guess for the model parameters
        """

        self.set_params_and_bounds(initial_parameters, fixed_params)

        # self._log(f"Starting calibration with initial parameters: {self.params}")
        # self._log(f"Bounds: {self._bounds}")

        print(f"Initial parameters: {self.params}")
        print(f"Bounds: {self._bounds}")
        print(f"Default bounds: {self.default_bounds}")

        def objective(params: list) -> float:
            try:
                self.params = params
                model_ivs = np.array(self.ivs(spot, k, t, r))
                return np.mean((model_ivs - market_ivs) ** 2)
            except Exception as e:
                # self._log(
                #     f"[RandSABR] Error in objective function: {e} with params: {params}",
                #     level="error",
                # )
                print(
                    f"[RandSABR] Error in objective function: {e} with params: {params}"
                )
                return np.inf

        result = basinhopping(
            objective,
            self.params,
            minimizer_kwargs={"method": "L-BFGS-B", "bounds": self._bounds},
            niter=n_iter,
            disp=verbose,
        )

        self.params = result.x.tolist()  # convert to list
        # self._log(f"Calibration completed. Calibrated parameters: {self.params}")
        # self.save_params()
