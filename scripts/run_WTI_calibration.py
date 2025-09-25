import os
from datetime import date
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd

from VolatilityRandomization import RandomizedModel
from VolatilityRandomization.models import SABR
from VolatilityRandomization.distributions import Gamma
from VolatilityRandomization.general.util import imply_volatility
from VolatilityRandomization.general.hagan import hagan_implied_volatility

"""
Runs the SABR randomization on the WTI option quotes in data/WTI.
We plot Standard SABR (Hagan IV) and the exact Randomized SABR (prices -> IV).
We also print and save LaTeX tables for SSE/MSE and for the randomized parameters.
"""

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Liberation Serif"]
plt.rcParams["text.usetex"] = True

if __name__ == "__main__":
    r = 0.05
    spot = 62.57
    today = date(2025, 9, 23)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 11))

    expiryDates = [
        date(2025, 11, 17),
        date(2025, 12, 16),
        date(2026, 1, 14),
        date(2026, 2, 17),
    ]
    filenames = [
        "WTI_20251117.csv",
        "WTI_20251216.csv",
        "WTI_20260114.csv",
        "WTI_20260217.csv",
    ]
    months = ["November", "December", "January", "February"]
    months_short = ["Nov", "Dec", "Jan", "Feb"]

    # Initialize data storage for RMSE and parameters
    rmse_data = {
        "RandSABR": [],
        "SABR": [],
    }
    parameters_data = {
        "beta": [],
        "alpha": [],
        "rho": [],
        "kappa": [],
        "theta": [],
        "var_gamma": [],
    }

    # Initial parameters for calibration
    sabr_initial_params = np.array([0.5, 0.5, 0, 1])
    rand_sabr_initial_params = np.array([0.5, 0.5, 0, 1, 1])

    for i, (eDate, filename, month) in enumerate(zip(expiryDates, filenames, months)):
        ax = axs.flatten()[i]

        # Load data
        data = pd.read_csv(
            f"data/WTI/{filename}", index_col=0).squeeze()

        # Calculate time to maturity
        t = (eDate - today).days / 365

        # Get strikes and implied volatilities
        k = np.array(data.index)
        iv = data.values
        print(f"{month} - Number of options: {len(k)}")
        print(f"Strikes range from {k.min()} to {k.max()}")
        print(f"{month} - Number of volatilities: {len(iv)}")
        print(
            f"Implied volatilities range from {iv.min():.4f} to {iv.max():.4f}")

        # Calibrate SABR model
        print(f"Calibrating SABR for {month}...")
        sabr_model = SABR()
        _ = sabr_model.calibrate(
            spot, k, t, r, iv, sabr_initial_params,
            fixed_params={"beta": 0.9}, n_iter=0, verbose=False
        )

        # Calibrate RandSABR model
        print(f"Calibrating RandSABR for {month}...")
        rand_sabr_model = RandomizedModel(
            model=SABR(),
            distribution=Gamma(),
            randomized_param="gamma",
        )
        _ = rand_sabr_model.calibrate(
            spot, k, t, r, iv, rand_sabr_initial_params,
            fixed_params={"beta": 0.9}, n_iter=0, verbose=False
        )
        
        # Store calibrated parameters
        parameters_data["beta"].append(rand_sabr_model.params[0])
        parameters_data["alpha"].append(rand_sabr_model.params[1])
        parameters_data["rho"].append(rand_sabr_model.params[2])
        parameters_data["kappa"].append(rand_sabr_model.params[3])
        parameters_data["theta"].append(rand_sabr_model.params[4])
        parameters_data["var_gamma"].append(rand_sabr_model.params[3] * rand_sabr_model.params[4]**2)
        
        # Compute errors on original strikes (like in the other script)
        sabr_prices_on_k = sabr_model.prices(spot, k, t, r)
        sabr_iv_on_k = imply_volatility(sabr_prices_on_k, spot, k, t, r, 0.4).flatten()
        
        rand_prices_on_k = rand_sabr_model.prices(spot, k, t, r)
        rand_iv_on_k = imply_volatility(rand_prices_on_k, spot, k, t, r, 0.4).flatten()
        
        # Calculate RMSE and store for table
        rmse_rand = np.sqrt(np.mean((iv - rand_iv_on_k) ** 2))
        rmse_sabr = np.sqrt(np.mean((iv - sabr_iv_on_k) ** 2))
        rmse_data["RandSABR"].append(rmse_rand)
        rmse_data["SABR"].append(rmse_sabr)

        # Create uniform grid for smooth plotting
        k_uni = np.linspace(k[0] / spot, k[-1] / spot, 100) * spot

        # Get model prices and implied volatilities
        sabr_prices = sabr_model.prices(spot, k_uni, t, r)
        rand_sabr_prices = rand_sabr_model.prices(spot, k_uni, t, r)
        
        # find the implied volatilities from the model prices
        sabr_iv = imply_volatility(sabr_prices, spot, k_uni, t, r, 0.4)
        rand_sabr_iv = imply_volatility(rand_sabr_prices, spot, k_uni, t, r, 0.4)
        
        # Plot market quotes
        ax.plot(
            k / spot,
            100 * iv,
            "x",
            label="Market Quote",
            markersize=7,
            color="sandybrown",
        )

        # Plot SABR fit
        ax.plot(
            k_uni / spot,
            100 * sabr_iv,
            label="SABR",
            marker="o",
            linestyle="dashdot",
            linewidth=1.5,
            color="green",
            markevery=5,
        )

        # Plot RandSABR fit
        ax.plot(
            k_uni / spot,
            100 * rand_sabr_iv,
            label="RandSABR",
            marker="^",
            linewidth=1.5,
            color="#00539C",
            markevery=5,
        )

        # Adjust Plot
        ax.grid()
        ax.set_xlabel("Strike (relative to ATM)", size=20)
        ax.set_ylabel("Implied Volatility [\\%]", size=20)
        ax.set_title(f"{month} (TTM {np.round(t, 2)}y)", size=20)
        ax.set_xlim([k[0] / spot, k[-1] / spot])

        l = ax.legend(prop={"size": 10}, fancybox=True, shadow=True)
        l.get_frame().set_edgecolor("black")

    plt.suptitle(
        f"WTI Option Chain {today.strftime('%Y-%m-%d')}", fontsize=22, fontweight="bold")
    plt.tight_layout()

    # Save the plot
    output_dir = "outputs/figs"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        f"{output_dir}/wti_implied_volatilities_with_models.png", dpi=300)
    plt.show()


    # --- Print and save the RMSE table in markdown style (like run_calibration.py) ---
    rmse_df = pd.DataFrame(rmse_data, index=months)
    rmse_df = rmse_df.T  # Models as rows, months as columns

    print("\n### RMSE Comparison Table (%)\n")
    print("| Model     | Nov      | Dec      | Jan      | Feb      |")
    print("|-----------|----------|----------|----------|----------|")
    for model_name, row in rmse_df.iterrows():
        print(f"| {model_name:<9} | "
              f"{row['November'] * 100:<8.4f} | "
              f"{row['December'] * 100:<8.4f} | "
              f"{row['January'] * 100:<8.4f} | "
              f"{row['February'] * 100:<8.4f} |")
    print()

    # Save RMSE table to CSV
    output_results_dir = "outputs/results"
    os.makedirs(output_results_dir, exist_ok=True)
    rmse_df.to_csv(f"{output_results_dir}/rmse_comparison_wti.csv")
    print(f"RMSE table saved to {output_results_dir}/rmse_comparison_wti.csv")
