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

"""
This script extends the functionality of the original run_randomized_sabr.py script by including
the calibration of both the RandSABR and SABR model parameters before running the analysis and plotting.
"""

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Liberation Serif"]
plt.rcParams["text.usetex"] = True
if __name__ == "__main__":
    r = 0.05
    spot = 5535
    today = date(2024, 7, 31)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 11))

    expiryDates = [
        date(2024, 8, 16),
        date(2024, 9, 20),
        date(2024, 10, 18),
        date(2024, 11, 15),
    ]
    months = ["August", "September", "October", "November"]

    # Initialize RMSE storage
    sse_data = {
        "Calibrated": [],
        "Pre-calibrated": [],
    }
    mse_data = {
        "Calibrated": [],
        "Pre-calibrated": [],
    }
    
    # initialize parameters storage for calibrated model
    parameters_data = {
        "beta": [],
        "alpha": [],
        "rho": [],
        "kappa": [],
        "theta": [],
        "var_gamma": [],
    }

    precalibrated_params_full = np.array(
        [
            [0.9, 0.33414235, -0.6907052, 1.70409171, 1.4284071],
            [0.9, 0.31866989, -0.6858314, 3.63637799, 0.48022903],
            [0.9, 0.31660493, -0.6753167, 2.89472607, 0.46295823],
            [0.9, 0.33841544, -0.68947279, 4.72176393, 0.28050375],
        ]
    )
    calibration_initial_params = np.mean(precalibrated_params_full, axis=0)
    calibration_initial_params = np.array([0.5, 0.5, 0, 1, 1])

    for (
        i,
        ax,
        eDate,
        month,
        precalibrated_params,
    ) in zip(
        range(4),
        axs.flatten(),
        expiryDates,
        months,
        precalibrated_params_full,
    ):
        data = pd.read_csv(f"data/{month}.csv", index_col=0).squeeze()

        t = (eDate - today).days / 365
        k, iv = np.array(data.index), data.values

        # Calibrate RandSABR parameters
        print(f"Calibrating RandSABR parameters for {month}...")
        calibrated_model = RandomizedModel(
            model=SABR(),
            distribution=Gamma(),
            randomized_param="gamma",
        )
        calibrated_model.calibrate(
            spot,
            k,
            t,
            r,
            iv,
            calibration_initial_params,
            fixed_params={"beta": 0.9},
            n_iter=20,
            verbose=True,
        )
        # print out the results and save parameters
        print(
            f"Calibrated RandSABR Parameters for {month}: {calibrated_model.params}"
        )

        parameters_data["beta"].append(calibrated_model.params[0])
        parameters_data["alpha"].append(calibrated_model.params[1])
        parameters_data["rho"].append(calibrated_model.params[2])
        parameters_data["kappa"].append(calibrated_model.params[3])
        parameters_data["theta"].append(calibrated_model.params[4])
        parameters_data["var_gamma"].append(calibrated_model.params[3] * calibrated_model.params[4]**2)

        
        # Create pre-calibrated RandSABR model
        precalibrated_model = RandomizedModel(
            model=SABR(),
            distribution=Gamma(),
            randomized_param="gamma",
            params=precalibrated_params,
        )

        # compute MSE and SSE (on the original strikes!)
        calibrated_prices_original = calibrated_model.prices(spot, k, t, r)
        precalibrated_prices_original = precalibrated_model.prices(spot, k, t, r)

        calibrated_ivs_original = imply_volatility(calibrated_prices_original, spot, k, t, r, 0.4).flatten()
        precalibrated_ivs_original = imply_volatility(precalibrated_prices_original, spot, k, t, r, 0.4).flatten()
        
        calibrated_mse = np.mean((iv - calibrated_ivs_original) ** 2)
        precalibrated_mse = np.mean((iv - precalibrated_ivs_original) ** 2)
        
        mse_data["Calibrated"].append(calibrated_mse)
        mse_data["Pre-calibrated"].append(precalibrated_mse)

        calibrated_sse = np.sum((iv - calibrated_ivs_original) ** 2)
        precalibrated_sse = np.sum((iv - precalibrated_ivs_original) ** 2)

        sse_data["Calibrated"].append(calibrated_sse)
        sse_data["Pre-calibrated"].append(precalibrated_sse)

        # For the plot we prefer a uniform grid
        k_uni = np.linspace(k[0] / spot, k[-1] / spot, 100) * spot

        # Get the prices and implied volatilities of the calibrated models
        calibrated_prices_plot = calibrated_model.prices(spot, k_uni, t, r)
        calibrated_ivs_plot = imply_volatility(calibrated_prices_plot, spot, k_uni, t, r, 0.4)

        precalibrated_prices_plot = precalibrated_model.prices(spot, k_uni, t, r)
        precalibrated_ivs_plot = imply_volatility(precalibrated_prices_plot, spot, k_uni, t, r, 0.4)

        # Plot the market data
        ax.plot(
            k / spot,
            100 * iv,
            "x",
            label="Market Quote",
            markersize=7,
            color="sandybrown",
        )
        # Plot the calibrated RandSABR
        ax.plot(
            k_uni / spot,
            100 * calibrated_ivs_plot,
            label="Calibrated RandSABR",
            marker="^",
            linewidth=1.5,
            color="blue",
            markevery=5,
        )
        # Plot the pre-calibrated RandSABR
        ax.plot(
            k_uni / spot,
            100 * precalibrated_ivs_plot,
            label="Pre-calibrated RandSABR",
            marker="^",
            linewidth=1.5,
            color="red",
            markevery=5,
        )
        # Adjust Plot
        ax.grid()
        ax.set_xlabel("Strike (relative to ATM)", size=20)
        ax.set_ylabel("Implied Volatility [%]", size=20)
        ax.set_title(f"{month} (TTM {np.round(t, 2)}y)", size=20)
        ax.set_xlim([k[0] / spot, k[-1] / spot])
        legend = ax.legend(prop={"size": 10}, fancybox=True, shadow=True)
        legend.get_frame().set_edgecolor("black")

    plt.suptitle(
        "SPX Option Chain 2024-07-31 New Calibration RandSABR",
        fontsize=22,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("outputs/figs/calibrated_vs_precalibrated_RandSABR.png", dpi=300)
    # plt.show()

    # --- Save the MSE and SSE results to LaTeX table ---
    latex_table = "\n\\begin{table}[h!]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{lcccc}\n"
    latex_table += "\\toprule\n"
    latex_table += "& \\multicolumn{2}{c}{Sum Squared Errors (SSE)} & \\multicolumn{2}{c}{Mean Squared Errors (MSE)} \\\\\n"
    latex_table += "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
    latex_table += "Expiry & Calibrated & Pre-calibrated & Calibrated & Pre-calibrated \\\\\n"
    latex_table += "\\midrule\n"
    months_short = ["Aug", "Sep", "Oct", "Nov"]
    for i in range(4):
        latex_table += (f"{months_short[i]} & "
                        f"{sse_data['Calibrated'][i]:.7f} & {sse_data['Pre-calibrated'][i]:.7f} & "
                        f"{mse_data['Calibrated'][i]:.2E} & {mse_data['Pre-calibrated'][i]:.2E} \\\\\n")
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Comparison of fitting errors between calibrated and pre-calibrated RandSABR models}\n"
    latex_table += "\\label{tab:error_comparison_calibration}\n"
    latex_table += "\\end{table}\n"
    
    # Save to file
    output_table_dir = "thesis/tables"
    os.makedirs(output_table_dir, exist_ok=True)
    with open(f"{output_table_dir}/MSE-SSE_Calibrated-vs-Precalibrated_RandSABR.tex", "w") as f:
        f.write(latex_table)
    # --- End LaTeX table output --- 
    
    # --- Save the calibrated parameters to LaTeX table ---
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
    latex_table += "\\caption{New calibrated parameters for the RandSABR model}\n"
    latex_table += "\\label{tab:calibration_results_new}\n"
    latex_table += "\\end{table}\n"
    
    # Save to file
    with open(f"{output_table_dir}/Calibrated_RandSABR_parameters.tex", "w") as f:
        f.write(latex_table)
    # --- End LaTeX table output ---
