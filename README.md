# Randomized Implied Volatility Parametrizations

## Installation

Clone this repository and install the library locally:

```bash
pip install -e .
```

This installs the package in editable mode, so any code changes are reflected immediately.
If you don’t need editable mode, simply run:

```
pip install .
```

Make sure your virtual environment is activated before running these commands.

## Quick Start

The library allows you to combine volatility models with probability distributions to build randomized implied volatility models.
For example, to create a randomized SABR model with the volatility-of-volatility parameter following a Gamma distribution:

```python
from VolatilityRandomization import RandomizedModel
from VolatilityRandomization.models import SABR
from VolatilityRandomization.distributions import Gamma

# Create randomized SABR model
rand_sabr = RandomizedModel(
    model=SABR(),
    distribution=Gamma(),
    randomized_param="gamma"
)

# Calibrate the model to market data
mse = rand_sabr.calibrate(
    spot=spot,
    strikes=strikes,
    maturity=maturity,
    rate=rate,
    market_ivs=market_implied_vols,
    initial_params=[0.9, 0.5, 0.0, 1.0, 1.0],  # [beta, alpha, rho, k, theta]
    fixed_params={"beta": 0.9}
)
```

This calibration optimizes both the model parameters and the distribution parameters simultaneously, enabling efficient fitting of randomized models without costly Monte Carlo simulation.

## Credits

This repository is based on the code from Zaugg, Perotti, and Grzelak in
_Volatility Parametrizations with Random Coefficients: Analytic Flexibility for Implied Volatility Surfaces_.
The original code can be found here: [NFZaugg/ImpliedVolatilityRandomization](https://github.com/NFZaugg/ImpliedVolatilityRandomization)
The codebase here has been reorganized into a modular framework with three key components:

1. Model classes – standard volatility models (e.g., SABR, Flat)

2. Distribution classes – statistical distributions for parameter randomization (e.g., Gamma, Normal, Uniform)

3. RandomizedModel – a wrapper combining models and distributions to build randomized variants

This design lets any model be paired with any distribution, making it straightforward to extend and test new randomizations.

## Architecture Overview

<img width="2506" height="1296" alt="framework_architecture" src="https://github.com/user-attachments/assets/c0209961-0d41-4c65-a0a0-c3ae6037d7e2" />

### Model Implementation

Each model class encapsulates the logic of a standard volatility model.

- `ivs`: compute implied volatilities

- `prices`: compute option prices (Black–Scholes)

- `calibrate`: fit model parameters to market data

For example, the SABR model includes the Hagan formula for option prices and calibration of (beta, alpha, rho, nu).

```python
class Model:
    param_names, default_bounds, params
    def parse_parameters(params): ...
    def validate_parameters(params, bounds): ...
    def ivs(spot, strikes, maturity, rate, params): ...
    def prices(spot, strikes, maturity, rate, params): ...
    def calibrate(...): ...

class SABR(Model):
    param_names = ["beta", "alpha", "rho", "gamma"]
    default_bounds = [(0.0, 1.0), (epsilon, None), (-1+epsilon, 1-epsilon), (epsilon, None)]
    def ivs(...):
        # Hagan formula
```

### Distribution Classes

Distribution classes define how parameters are randomized, handling:

- Gram matrix computation (moments)

- Quadrature nodes and weights (Gaussian quadrature integration)

- Validation of parameters

Currently implemented: Gamma, Normal, Uniform.

```python
class Distribution:
    param_names, bounds, params
    def get_gram_matrix(n, params): ...
    def collocation_points(n, params): ...

class Gamma(Distribution):
    param_names = ["shape", "scale"]
    bounds = [(epsilon, None), (epsilon, None)]
    def get_gram_matrix(n, params): ...
```

### RandomizedModel

The RandomizedModel class glues everything together:

- Combines a model and a distribution

- Specifies which parameter(s) to randomize

- Calibrates both model and distribution parameters

- Uses Gaussian quadrature for efficient integration

```python
class RandomizedModel:
    model, distribution, randomized_param, params, n_quad_points
    def generate_parameter_sets(): ...
    def prices(spot, strikes, maturity, rate): ...
    def ivs(spot, strikes, maturity, rate): ...
    def calibrate(...): ...
```

<p align="center">
  <img width="782" height="558" alt="randomized-iv-flowchart" src="https://github.com/user-attachments/assets/d6063b9d-9970-4b87-ba79-33ed28faccd1" />
</p>

