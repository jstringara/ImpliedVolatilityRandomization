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
Runs the SABR randomization on the SPX option quotes in the Data folder. 
We plot both the exact randomization values, as well as the 6th order expansion. 
We see that the 6th order is not sufficient for this case, as the approximation diverges from the exact solution.

However, the fit of the randomized SABR is significantly better due to the extra flexiblity.

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
    sabr_params = np.array(
        [
            [0.9, 0.33795777, -0.54409678, 3.94972169],
            [0.9, 0.32457636, -0.59188021, 2.16977728],
            [0.9, 0.3280161, -0.47201946, 1.65175483],
            [0.9, 0.34742342, -0.56569068, 1.5388523],
        ]
    )
    # Prepare to collect errors for LaTeX table
    months_short = ["Aug", "Sep", "Oct", "Nov"]
    mse_sabr_list = []
    mse_rand_list = []
    sse_sabr_list = []
    sse_rand_list = []

    for i, ax, eDate, month, param, sabr_params in zip(
        range(4), axs.flatten(), expiryDates, months, rand_sabr_params, sabr_params
    ):
        data = pd.read_csv(f"data/{month}.csv", index_col=0).squeeze()

        t = (eDate - today).days / 365
        k, iv = np.array(data.index), data.values
        randomized_sabr = RandomizedModel(
            model=SABR(),
            distribution=Gamma(),
            randomized_param="gamma",
            params=param,
        )

        # For the plot we prefer a uniform grid
        k_uni = np.linspace(k[0] / spot, k[-1] / spot, 100) * spot

        # Get the randomized prices and expansion IVS
        randomized_prices = randomized_sabr.prices(spot, k_uni, t, r)
        # Plotting
        ax.plot(
            k / spot,
            100 * iv,
            "x",
            label="Market Quote",
            markersize=7,
            color="sandybrown",
        )
        ax.plot(
            k_uni / spot,
            100 * imply_volatility(randomized_prices, spot, k_uni, t, r, 0.4),
            label="Randomized SABR",
            marker="^",
            linewidth=1.5,
            color="#00539C",
            markevery=5,
        )

        ax.plot(
            k_uni / spot,
            100
            * hagan_implied_volatility(k_uni, t, spot * np.exp(r * t), *sabr_params),
            label="Regular SABR",
            marker="o",
            linestyle="dashdot",
            linewidth=1.5,
            color=plt.cm.viridis(0.5),
            markevery=5,
        )
        # --- Compute MSE and SSE on original strikes ---
        # Classical SABR IVs on original strikes
        sabr_iv = hagan_implied_volatility(k, t, spot * np.exp(r * t), *sabr_params)
        # RandSABR prices and IVs on original strikes
        rand_prices = randomized_sabr.prices(spot, k, t, r)
        rand_iv = imply_volatility(rand_prices, spot, k, t, r, 0.4).flatten()

        # Compute errors
        mse_sabr = np.mean((iv - sabr_iv) ** 2)
        sse_sabr = np.sum((iv - sabr_iv) ** 2)
        mse_rand = np.mean((iv - rand_iv) ** 2)
        sse_rand = np.sum((iv - rand_iv) ** 2)

        mse_sabr_list.append(mse_sabr)
        mse_rand_list.append(mse_rand)
        sse_sabr_list.append(sse_sabr)
        sse_rand_list.append(sse_rand)

        print(f"\n{month} (TTM {np.round(t, 2)}y):")
        print(f"  SABR      MSE: {mse_sabr:.6f}, SSE: {sse_sabr:.6f}")
        print(f"  RandSABR  MSE: {mse_rand:.6f}, SSE: {sse_rand:.6f}")
        # --- End error computation ---

        # Adjust Plot
        ax.grid()
        ax.set_xlabel("Strike (relative to ATM) ", size=20)
        ax.set_ylabel("Implied Volatility [%]", size=20)
        ax.set_title(f"{month} (TTM {np.round(t, 2)}y)", size=20)
        ax.set_xlim([k[0] / spot, k[-1] / spot])
        l = ax.legend(prop={"size": 10}, fancybox=True, shadow=True)
        l.get_frame().set_edgecolor("black")

    import os

    plt.suptitle("SPX Option Chain 2024-07-31", fontsize=22, fontweight="bold")
    plt.tight_layout()
    output_dir = "outputs/figs"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/randomized_sabr_with_data.png", dpi=300)
    plt.show()
