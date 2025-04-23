from datetime import date
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping
from randomizations.rand_sabr import RandSABR
from general.util import imply_volatility

"""
This script extends the functionality of the original run_randomized_sabr.py script by including
the calibration of the RandSABR model parameters before running the analysis and plotting.
"""


def calibrate_rand_sabr(spot, k, t, r, market_ivs, initial_parameters, pre_calibrated_parameters, n_col_points=2):
    """
    Calibrates the RandSABR model parameters by minimizing the error between the model's
    implied volatilities and the observed market implied volatilities.
    """
    def objective(params):
        model = RandSABR(params_rand=params, n_col_points=n_col_points)
        try:
            model_ivs = model.ivs(spot, k, t, r)
            error = np.mean((np.array(model_ivs) - market_ivs)
                            ** 2)  # Normalize error
            return error
        except Exception as e:
            print("Error during model evaluation:", e)
            return np.inf

    # Parameter bounds
    bounds = [
        (1e-6, None),  # alpha > 0
        (0, 1),        # 0 <= beta <= 1
        (-1+1e-6, 1-1e-6),  # rho in (-1, 1)
        (1e-6, None),  # gamma > 0
        (1e-6, None),  # scale > 0
    ]
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
    result = basinhopping(objective, initial_parameters,
                          minimizer_kwargs=minimizer_kwargs, niter=10)
    calibrated_params = result.x

    # validate the calibration
    calibrated_error = objective(calibrated_params)
    print(
        f"objective function value at calibrated parameters: {calibrated_error}")
    pre_calibrated_error = objective(pre_calibrated_parameters)
    print(
        f"objective function value at pre-calibrated parameters: {pre_calibrated_error}"
    )

    return calibrated_params


rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Liberation Serif"]
plt.rcParams["text.usetex"] = True
if __name__ == "__main__":
    r = 0.05
    spot = 5535
    today = date(2024, 7, 31)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 11))

    expiryDates = [date(2024, 8, 16), date(2024, 9, 20),
                   date(2024, 10, 18), date(2024, 11, 15)]
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

    for i, ax, eDate, month, pre_calibrated_params in zip(
        range(4), axs.flatten(), expiryDates, months, rand_sabr_params
    ):
        data = pd.read_csv(f"Data/{month}.csv", index_col=0).squeeze()

        t = (eDate - today).days / 365
        k, iv = np.array(data.index), data.values

        # Calibrate RandSABR parameters
        print(f"Calibrating parameters for {month}...")
        calibrated_randomized_sabr = RandSABR()
        calibrated_randomized_sabr.calibrate(
            spot, k, t, r, iv, rand_sabr_initial_params, n_iter=10)
        calibrated_params = calibrated_randomized_sabr.params_rand
        print(f"Calibrated Parameters for {month}: {calibrated_params}")
        
        pre_calibrated_randomized_sabr = RandSABR(
            params_rand=pre_calibrated_params, n_col_points=2)

        # For the plot we prefer a uniform grid
        k_uni = np.linspace(k[0] / spot, k[-1] / spot, 100) * spot

        # Get the randomized prices and expansion IVS
        calibrated_randomized_prices = calibrated_randomized_sabr.prices(
            spot, k_uni, t, r)
        calibrated_randomized_ivs = calibrated_randomized_sabr.ivs(
            spot, k_uni, t, r)
        pre_calibrated_randomized_prices = pre_calibrated_randomized_sabr.prices(
            spot, k_uni, t, r)
        pre_calibrated_randomized_ivs = pre_calibrated_randomized_sabr.ivs(
            spot, k_uni, t, r)
        print(
            f"Pre-calibrated Parameters for {month}: {pre_calibrated_params}")
        print(f"Calibrated Parameters for {month}: {calibrated_params}")
        print(
            f"Difference from pre-calibrated parameters: {np.abs(calibrated_params - pre_calibrated_params)}")

        # Plotting
        ax.plot(k / spot, 100 * iv, "x", label="Market Quote",
                markersize=7, color="sandybrown")
        # plot the calibrated randomized SABR
        ax.plot(
            k_uni / spot,
            100 * imply_volatility(calibrated_randomized_prices,
                                   spot, k_uni, t, r, 0.4),
            label="Randomized SABR",
            marker="^",
            linewidth=1.5,
            color="#00539C",
            markevery=5,
        )
        ax.plot(
            k_uni / spot,
            100 * np.array(calibrated_randomized_ivs),
            label="Randomized SABR (6th-order approximation)",
            marker="*",
            linewidth=1.5,
            color="purple",
            markevery=5,
        )
        # plot the pre-calibrated randomized SABR
        ax.plot(
            k_uni / spot,
            100 *
            imply_volatility(pre_calibrated_randomized_prices,
                             spot, k_uni, t, r, 0.4),
            label="Pre-calibrated Randomized SABR",
            marker="o",
            linewidth=1.5,
            color="#FF6F61",
            markevery=5,
        )
        ax.plot(
            k_uni / spot,
            100 * np.array(pre_calibrated_randomized_ivs),
            label="Pre-calibrated Randomized SABR (6th-order approximation)",
            marker="s",
            linewidth=1.5,
            color="green",
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

    plt.suptitle("SPX Option Chain 2024-07-31 with Calibrated RandSABR",
                 fontsize=22, fontweight="bold")
    plt.tight_layout()
    plt.savefig("Plots/randomized_sabr_with_calibration.png", dpi=300)
    plt.show()
