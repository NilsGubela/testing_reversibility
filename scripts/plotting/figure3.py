import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import norm
from util import * 

ROOT = Path(__file__).resolve().parents[2]
out_dir = ROOT / "results" / "figure3"

fig_dir = ROOT / "figs"

files = [out_dir / "result_100000_500_1p0.csv",out_dir / "result_100000_5000_1p0.csv", out_dir / "result_100000_10000_1p0.csv"]

all_res = []
for file in files:
    tmp_df = pd.read_csv(file)
    all_res.append(tmp_df["stat"])

x = np.linspace(-5, 5, 500)
pdf = norm.pdf(x)#(1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(3,3,figsize=(single_half_col, 4))
    for i, res in enumerate(all_res):
        # histogram
        ax[0,i].hist(res, bins = 100, density = True, color = "steelblue")
        ax[0,i].plot(x, pdf, label="N(0,1) pdf", color = "red")
        ax[0,i].set_xlabel(r"$\hat{d}_{ijk}$")

        # QQ plot
        stats.probplot(res, dist="norm", sparams=(0, 1), plot=ax[1,i])
        points_line = ax[1,i].lines[0]
        ref_line = ax[1,i].lines[1]
        # Customize points (marker color, size, transparency)
        points_line.set_markerfacecolor("steelblue")
        points_line.set_markeredgecolor("steelblue")
        points_line.set_markersize(3)
        points_line.set_alpha(0.6)
        # Customize reference line
        ref_line.set_color("red")
        ref_line.set_linewidth(2)
        ax[1,i].set_title("")
        ax[1,i].set_ylabel("")
        ax[1,i].set_xlabel("")

        # cumulative dist 
        res = np.asarray(res)
        res = res[~np.isnan(res)]
        # One-sample KS test vs standard normal N(0,1)
        ks = stats.kstest(res, cdf="norm", args=(0, 1))
        print(i, "KS-statistic:", np.round(ks.statistic,4 ))
        res_sorted = np.sort(res)
        n = res_sorted.size
        # ECDF
        ecdf_y = np.arange(1, n + 1) / n
        # Standard normal CDF on a grid
        #x = np.linspace(res_sorted[0], res_sorted[-1], 500)
        cdf_y = norm.cdf(x)  # N(0,1)
        ax[2,i].step(res_sorted, ecdf_y, where="post", label="ECDF(res)", color="black")
        ax[2,i].plot(x, cdf_y, label="CDF N(0,1)", color="red", linewidth=2, linestyle = "dashed")

        ax[2,i].set_xlabel("x")
        ax[1,i].set_xlabel("Theoretical quantiles")
        # beauty work
        if i > 0:
            for k in range(3):
                ax[k,i].set_yticklabels([])

    ax[2,0].set_ylabel("Cumulative probability")    
    ax[1,0].set_ylabel("Sample quantiles")
    ax[0,0].set_ylabel("Density")

    plt.tight_layout()

    #plt.savefig(fig_dir / "figure3.svg")
    plt.savefig(fig_dir / "figure3.pdf")
    plt.savefig(fig_dir / "figure3.png")
    plt.show()