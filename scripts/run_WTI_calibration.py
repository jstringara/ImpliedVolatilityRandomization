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

    # Initialize data storage for LaTeX tables
    sse_data = {
        "SABR": [],
        "RandSABR": [],
    }
    mse_data = {
        "SABR": [],
        "RandSABR": [],
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
            fixed_params={"beta": 0.9}, n_iter=10, verbose=False
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
        
        # Calculate SSE and MSE
        mse_sabr = np.mean((iv - sabr_iv_on_k) ** 2)
        sse_sabr = np.sum((iv - sabr_iv_on_k) ** 2)
        mse_rand = np.mean((iv - rand_iv_on_k) ** 2)
        sse_rand = np.sum((iv - rand_iv_on_k) ** 2)
        
        # Store errors for tables
        mse_data["SABR"].append(mse_sabr)
        mse_data["RandSABR"].append(mse_rand)
        sse_data["SABR"].append(sse_sabr)
        sse_data["RandSABR"].append(sse_rand)

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

    # --- Print the table for MSE and SSE comparison ---
    print("\nComparison of fitting errors between standard and randomized SABR models for WTI:\n")
    for i in range(4):
        print(f"{months_short[i]}: ")
        print(f"  Standard SABR - MSE: {mse_data['SABR'][i]:.2E}, SSE: {sse_data['SABR'][i]:.7f}")
        print(f"  Randomized SABR - MSE: {mse_data['RandSABR'][i]:.2E}, SSE: {sse_data['RandSABR'][i]:.7f}\n")

    # --- Generate LaTeX tables ---
    output_table_dir = "outputs/tables"
    os.makedirs(output_table_dir, exist_ok=True)

    # LaTeX table for MSE and SSE comparison
    latex_table = "\n\\begin{table}[h!]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{lcccc}\n"
    latex_table += "\\toprule\n"
    latex_table += "& \\multicolumn{2}{c}{Sum Squared Errors (SSE)} & \\multicolumn{2}{c}{Mean Squared Errors (MSE)} \\\\\n"
    latex_table += "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
    latex_table += "Expiry & Standard SABR & Randomized SABR & Standard SABR & Randomized SABR \\\\\n"
    latex_table += "\\midrule\n"
    for i in range(4):
        latex_table += (f"{months_short[i]} & "
                        f"{sse_data['SABR'][i]:.7f} & {sse_data['RandSABR'][i]:.7f} & "
                        f"{mse_data['SABR'][i]:.2E} & {mse_data['RandSABR'][i]:.2E} \\\\\n")
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Comparison of fitting errors between standard and randomized SABR models for WTI options}\n"
    latex_table += "\\label{tab:error_comparison_wti}\n"
    latex_table += "\\end{table}\n"

    with open(f"{output_table_dir}/MSE-SSE_SABR-Rand_WTI.tex", "w") as f:
        f.write(latex_table)

    # LaTeX table for RandSABR parameters
    latex_table = "\n\\begin{table}[h!]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{lccccccc}\n"
    latex_table += "\\toprule\n"
    latex_table += "Expiry & $\\beta$ & $\\alpha$ & $\\rho$ & $k$ & $\\theta$ & Var($\\Gamma$) \\\\\n"
    latex_table += "\\midrule\n"
    for i in range(4):
        latex_table += (f"{months_short[i]} & "
                        f"{parameters_data['beta'][i]:.1f} & {parameters_data['alpha'][i]:.3f} & {parameters_data['rho'][i]:.3f} & "
                        f"{parameters_data['kappa'][i]:.3f} & {parameters_data['theta'][i]:.3f} & {parameters_data['var_gamma'][i]:.3f} \\\\\n")
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Calibrated parameters for the randomized SABR model on WTI options}\n"
    latex_table += "\\label{tab:wti_rand_sabr_parameters}\n"
    latex_table += "\\end{table}\n"
    
    with open(f"{output_table_dir}/RandSABR_parameters_WTI.tex", "w") as f:
        f.write(latex_table)

    print(f"\nLaTeX tables saved to {output_table_dir}/")
    print("- MSE-SSE_SABR-Rand_WTI.tex")
    print("- RandSABR_parameters_WTI.tex")