import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import ClassVar, Optional, Union, Self
from pydantic import BaseModel, Field, model_validator
from scipy.optimize import basinhopping

from general.collocation import get_col_points
from general.hagan import hagan_implied_volatility
from general.expansions import get_sigma_0, get_sigma_approx
from general.util import black76


class RandSABR(BaseModel):
    """
    Random SABR model with randomized gamma parameter (vol of vol),
    modeled via a Gamma distribution with shape and scale parameters.
    """

    # Parameter names
    param_names: ClassVar[list[str]] = ["beta", "alpha", "rho", "gamma", "scale"]
    non_randomized_params_end: ClassVar[int] = 3

    # Bounds for the parameters
    epsilon: ClassVar[float] = 1e-5
    default_bounds: ClassVar[list[tuple[float, Optional[float]]]] = [
        (0.0, 1.0),  # beta in [0,1]
        (epsilon, None),  # alpha > 0
        (-1 + epsilon, 1 - epsilon),  # rho in (-1,1)
        (epsilon, None),  # gamma > 0
        (epsilon, None),  # scale > 0
    ]

    # Parameters
    params: Optional[list[float]] = Field(default_factory=list)
    fixed_params: dict[str, float] = Field(default_factory=dict)
    bounds: list[tuple[float, Optional[float]]] = Field(default_factory=list)

    n_col_points: int = 2

    # Logging and parameter storage
    log_file: ClassVar[str] = os.path.join(
        os.path.dirname(__file__), "Logs", "RandSABR.log"
    )
    params_file: ClassVar[str] = os.path.join(
        os.path.dirname(__file__), "Calibrations", "RandSABR_params.json"
    )

    # Initialize logger
    logger: ClassVar[logging.Logger] = logging.getLogger("RandSABR")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    def _log(self, message: str, level: str = "info") -> None:
        """
        Log a message using the logging library.
        """
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)
        print(message)  # Also print to the terminal

    def show_logs(self) -> None:
        """
        Display the contents of the log file.
        """
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as log:
                print(log.read())
        else:
            print("No logs available.")

    def save_params(self) -> None:
        """
        Save the calibrated parameters to a JSON file with a timestamp.
        """
        if not self.params:
            raise ValueError("No parameters to save.")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        serialized_params = self.to_dict()
        
        if not os.path.exists(os.path.dirname(self.params_file)):
            os.makedirs(os.path.dirname(self.params_file))

        if os.path.exists(self.params_file):
            with open(self.params_file, "r") as f:
                data = json.load(f)
        else:
            data = {}

        data[timestamp] = serialized_params

        with open(self.params_file, "w") as f:
            json.dump(data, f, indent=4)

        self._log(
            f"Calibrated parameters saved to {self.params_file} under timestamp {timestamp}."
        )

    def load_params(self, timestamp: str) -> None:
        """
        Load calibrated parameters from the JSON file by timestamp.
        """
        if not os.path.exists(self.params_file):
            raise FileNotFoundError("No parameter file found.")

        with open(self.params_file, "r") as f:
            data = json.load(f)

        if timestamp is None:
            # Load the most recent parameters
            timestamp = sorted(data.keys())[-1]

        if timestamp not in data:
            raise ValueError(f"No parameters found for timestamp {timestamp}.")

        self.params = self.params_dict_to_list(data[timestamp])
        self._log(f"Loaded parameters for timestamp {timestamp}: {self.params}")

    @classmethod
    def _parse_params(
        cls, params: Union[np.ndarray, tuple, dict, list, None]
    ) -> list[float]:
        """
        Parse and convert the input parameters into a list of floats.
        Supports numpy arrays, tuples, dictionaries, lists, or None.
        """
        if params is None:
            return []

        if isinstance(params, np.ndarray):
            if len(params) != len(cls.param_names):
                raise ValueError(
                    f"Expected {len(cls.param_names)} parameters, got {len(params)}"
                )
            return params.tolist()

        if isinstance(params, tuple):
            if len(params) != len(cls.param_names):
                raise ValueError(
                    f"Expected {len(cls.param_names)} parameters, got {len(params)}"
                )
            return list(params)

        if isinstance(params, dict):
            if len(params) != len(cls.param_names):
                raise ValueError(
                    f"Expected {len(cls.param_names)} parameters, got {len(params)}"
                )
            if params.keys() != set(cls.param_names):
                raise ValueError(
                    f"Expected parameters {cls.param_names}, got {params.keys()}"
                )
            return [params[name] for name in cls.param_names]

        if isinstance(params, list):
            if len(params) != len(cls.param_names):
                raise ValueError(
                    f"Expected {len(cls.param_names)} parameters, got {len(params)}"
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

    @model_validator(mode="after")
    def _validate_params(self) -> Self:
        self._validate_params_with_bounds(self.params, self.bounds)
        self._validate_fixed_params(self.fixed_params)

        return self

    def _validate_params_with_bounds(
        self, params: list[float], bounds: list[tuple[float, Optional[float]]]
    ) -> Self:
        """
        Validate that the parameters are within the specified bounds.
        """
        # short circuit if not params are set
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

    @property
    def gamma_params(self) -> list[float]:
        """
        Returns the parameters of the gamma distribution.
        """
        if len(self.params) == 0:
            raise ValueError("Parameters not set.")
        return self.params[self.non_randomized_params_end :]

    @property
    def non_randomized_params(self) -> list[float]:
        """
        Returns the non-randomized parameters of the model.
        """
        if len(self.params) == 0:
            raise ValueError("Parameters not set.")
        return self.params[: self.non_randomized_params_end]

    def set_params_and_bounds(
        self,
        initial_parameters: Union[np.ndarray, tuple, dict[str, float], list[float]],
    ) -> None:
        """
        Set the parameters and bounds, fixing any parameters as needed.
        """
        # validate the input parameters
        initial_parameters = self._parse_params(initial_parameters)

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
        Calibrates the RandSABR model parameters using market data and saves them to params.
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
                self._log(
                    f"[RandSABR] Error in objective function: {e} with params: {params}",
                    level="error",
                )
                return np.inf

        self.set_params_and_bounds(initial_parameters)

        self._log(f"Starting calibration with initial parameters: {self.params}")
        self._log(f"Bounds: {self.bounds}")

        result = basinhopping(
            objective,
            self.params,
            minimizer_kwargs={"method": "L-BFGS-B", "bounds": self.bounds},
            niter=n_iter,
            disp=verbose,
        )

        self.params = result.x.tolist()  # convert to list
        self._log(f"Calibration completed. Calibrated parameters: {self.params}")
        self.save_params()

    def prices(self, spot: float, k: np.ndarray, t: float, r: float) -> np.ndarray:
        """
        Return model prices for a set of strikes.
        """
        return self._mixed_hagan_prices(spot, k, t, r)

    def ivs(self, spot: float, k: np.ndarray, t: float, r: float) -> np.ndarray:
        """
        Return model implied volatilities for a set of strikes.
        """
        return self._mixed_hagan_ivs(spot, k, t, r)

    def _generate_param_sets(self) -> tuple[np.ndarray, list[list[float]]]:
        """
        Generate parameter sets by collocating on the gamma randomization.
        """
        weights, col_points = get_col_points(
            self.n_col_points, self.gamma_params, "gamma"
        )
        param_sets = [(*self.non_randomized_params, c) for c in col_points]
        return weights, param_sets

    def _mixed_hagan_ivs(
        self, spot: float, k: np.ndarray, t: float, r: float
    ) -> list[float]:
        """
        Compute implied volatilities by mixing over randomization.
        """
        weights, params = self._generate_param_sets()
        model_ivs = [
            np.atleast_1d(hagan_implied_volatility(k, t, spot * np.exp(r * t), *p))
            for p in params
        ]

        ivs_mixed = []
        for strike_idx, strike in enumerate(k):
            m = np.log(spot / strike) + r * t
            ivs_at_strike = np.array([ivs[strike_idx] for ivs in model_ivs])
            sigma0 = get_sigma_0(t, weights, ivs_at_strike)
            ivs_mixed.append(get_sigma_approx(m, t, sigma0, weights, ivs_at_strike))
        return ivs_mixed

    def _mixed_hagan_prices(
        self, spot: float, k: np.ndarray, t: float, r: float
    ) -> np.ndarray:
        """
        Compute prices by mixing over randomization.
        """
        weights, params = self._generate_param_sets()
        f = spot * np.exp(r * t)
        prices_for_params = [
            black76(f, k, t, r, hagan_implied_volatility(k, t, f, *p)) for p in params
        ]
        return (weights[:, np.newaxis] * prices_for_params).sum(axis=0)
