from datetime import date
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping
from randomizations.rand_sabr import RandSABR
from randomizations.sabr import SABR  # Import the new SABR class
from general.util import imply_volatility
from general.hagan import hagan_implied_volatility

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
    rand_sabr_params = np.array(
        [
            [0.9, 0.33414235, -0.6907052, 1.70409171, 1.4284071],
            [0.9, 0.31866989, -0.6858314, 3.63637799, 0.48022903],
            [0.9, 0.31660493, -0.6753167, 2.89472607, 0.46295823],
            [0.9, 0.33841544, -0.68947279, 4.72176393, 0.28050375],
        ]
    )
    rand_sabr_initial_params = np.mean(rand_sabr_params, axis=0)
    rand_sabr_initial_params = np.array([0.5, 0.5, 0, 1, 1])
    sabr_params = np.array(
        [
            [0.9, 0.33795777, -0.54409678, 3.94972169],
            [0.9, 0.32457636, -0.59188021, 2.16977728],
            [0.9, 0.3280161, -0.47201946, 1.65175483],
            [0.9, 0.34742342, -0.56569068, 1.5388523],
        ]
    )
    sabr_initial_params = np.mean(rand_sabr_params, axis=0)
    sabr_initial_params = np.array([0.5, 0.5, 0, 1])

    for (
        i,
        ax,
        eDate,
        month,
        pre_calibrated_rand_sabr_params,
        pre_calibrated_sabr_params,
    ) in zip(
        range(4),
        axs.flatten(),
        expiryDates,
        months,
        rand_sabr_params,
        sabr_params,
    ):
        data = pd.read_csv(f"Data/{month}.csv", index_col=0).squeeze()

        t = (eDate - today).days / 365
        k, iv = np.array(data.index), data.values

        # Calibrate RandSABR parameters
        print(f"Calibrating RandSABR parameters for {month}...")
        calibrated_rand_sabr = RandSABR(fixed_params={"beta": 0.9})
        calibrated_rand_sabr.calibrate(
            spot, k, t, r, iv, rand_sabr_initial_params, n_iter=10, verbose=True
        )

        # Calibrate SABR parameters
        print(f"Calibrating SABR parameters for {month}...")
        calibrated_sabr = SABR(fixed_params={"beta": 0.9})
        calibrated_sabr.calibrate(spot, k, t, r, iv, sabr_initial_params, n_iter=10, verbose=True)

        # For the plot we prefer a uniform grid
        k_uni = np.linspace(k[0] / spot, k[-1] / spot, 100) * spot

        # Get the prices and implied volatilities of the calibrated models
        calibrated_rand_sabr_prices = calibrated_rand_sabr.prices(spot, k_uni, t, r)
        calibrated_sabr_prices = calibrated_sabr.prices(spot, k_uni, t, r)

        # Get the prices and implied volatilities of the pre-calibrated models
        pre_calibrated_rand_sabr = RandSABR(params=pre_calibrated_rand_sabr_params)
        pre_calibrated_sabr = SABR(params=pre_calibrated_sabr_params)

        pre_calibrated_rand_sabr_prices = pre_calibrated_rand_sabr.prices(
            spot, k_uni, t, r
        )
        pre_calibrated_sabr_prices = pre_calibrated_sabr.prices(spot, k_uni, t, r)

        print(
            f"Calibrated RandSABR Parameters for {month}: {calibrated_rand_sabr.params}"
        )
        print(f"Calibrated SABR Parameters for {month}: {calibrated_sabr.params}")

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
            100 * imply_volatility(calibrated_rand_sabr_prices, spot, k_uni, t, r, 0.4),
            label="Calibrated RandSABR",
            marker="^",
            linewidth=1.5,
            color="blue",
            markevery=5,
        )
        # Plot the calibrated SABR
        ax.plot(
            k_uni / spot,
            100 * imply_volatility(calibrated_sabr_prices, spot, k_uni, t, r, 0.4),
            label="Calibrated SABR",
            marker="o",
            linestyle="dashdot",
            linewidth=1.5,
            color="green",
            markevery=5,
        )
        # Plot the pre-calibrated RandSABR
        ax.plot(
            k_uni / spot,
            100
            * imply_volatility(pre_calibrated_rand_sabr_prices, spot, k_uni, t, r, 0.4),
            label="Pre-calibrated RandSABR",
            marker="^",
            linewidth=1.5,
            color="red",
            markevery=5,
        )
        # Plot the pre-calibrated SABR
        ax.plot(
            k_uni / spot,
            100 * imply_volatility(pre_calibrated_sabr_prices, spot, k_uni, t, r, 0.4),
            label="Pre-calibrated SABR",
            marker="o",
            linestyle="--",
            linewidth=1.5,
            color="orange",
            markevery=5,
        )

        # Adjust Plot
        ax.grid()
        ax.set_xlabel("Strike (relative to ATM)", size=20)
        ax.set_ylabel("Implied Volatility [%]", size=20)
        ax.set_title(f"{month} (TTM {np.round(t, 2)}y)", size=20)
        ax.set_xlim([k[0] / spot, k[-1] / spot])
        l = ax.legend(prop={"size": 10}, fancybox=True, shadow=True)
        l.get_frame().set_edgecolor("black")

    plt.suptitle(
        "SPX Option Chain 2024-07-31 with Calibrated RandSABR and SABR",
        fontsize=22,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("Plots/randomized_sabr_with_calibration.png", dpi=300)
    plt.show()
