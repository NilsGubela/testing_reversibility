import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.lines import Line2D
from util import * 

ROOT = Path(__file__).resolve().parents[2]
out_dir = ROOT / "results" / "figure1"

fig_dir = ROOT / "figs"


def plot_b_s_qs(data, ax, row, t, col_index):
    ax[row,0].scatter(data[t], data["Bowker"], s = 1, color = "steelblue")
    ax[row,0].axhline(chi2.ppf(0.95, 6), linestyle = "dashed", color = "k")
    ax[row,0].set_xlabel(r"$t$")
    ax[row,0].set_ylabel(r"$S^2_B$", rotation=0)

    ax[row,1].scatter(data[t], data["Stuart"], s = 1, color = "tab:purple")
    ax[row,1].axhline(chi2.ppf(0.95, 3), linestyle = "dashed", color = "k")
    ax[row,1].set_xlabel(r"$t$")
    ax[row,1].set_ylabel("Stuart test statistic")
    ax[row,1].set_ylabel(r"$S^2_S$", rotation=0)

    ax[row,2].scatter(data[t], data["QS"], s = 1, color = "tab:orange")
    ax[row,2].axhline(norm.ppf(0.975), linestyle = "dashed", color = "k")
    ax[row,2].axhline(norm.ppf(0.025), linestyle = "dashed", color = "k")
    ax[row,2].set_xlabel(r"$t$")
    ax[row,2].set_ylabel(r"$S^2_{QS}$", rotation=0)

    colors = ["grey","grey","grey","grey","grey"]
    colors[col_index] = "forestgreen"

    values = [0,1,2,3,4]
    bc = np.bincount(data["y"])
    counts = bc[values]
    bars = ax[row,3].barh(values, counts/np.sum(counts) * 100, color = colors)
    ax[row,3].bar_label(bars, padding=0)
    ax[row,3].set_xlabel("Percentage")

    if t == "t2":
        ax[row,0].set_xlabel(r"$t_2$")
        ax[row,1].set_xlabel(r"$t_2$")
        ax[row,2].set_xlabel(r"$t_2$")
    ax[row,3].set_yticks([0,1,2,3,4],labels)


# combine them all 

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(4,4,figsize=(6.29, 4*1.3),)
                           #layout="constrained")

    data = pd.read_csv(out_dir / "result_nonstationary_reversible.csv")
    data = evaluate_tests(data)
    data = add_decision(data)
    data.loc[data["Bowker"].isna(),"final_decision"] = np.nan
    data["y"] = data["final_decision"].map(y_map)
    plot_b_s_qs(data, ax, 0, "t",2)

    data = pd.read_csv(out_dir / "result_stationary_nonreversible.csv")
    data = evaluate_tests(data)
    data = add_decision(data)
    data.loc[data["Bowker"].isna(),"final_decision"] = np.nan
    data["y"] = data["final_decision"].map(y_map)
    plot_b_s_qs(data, ax, 1, "t",3)

    data = pd.read_csv(out_dir / "result_nonstationary_reversible_cherry.csv")
    data = evaluate_tests(data)
    data = add_decision(data)
    data.loc[data["Bowker"].isna(),"final_decision"] = np.nan
    data["y"] = data["final_decision"].map(y_map)
    plot_b_s_qs(data, ax, 2, "t2",2)

    data = pd.read_csv(out_dir / "result_stationary_nonreversible_cherry.csv")
    data = evaluate_tests(data)
    data = add_decision(data)
    data.loc[data["Bowker"].isna(),"final_decision"] = np.nan
    data["y"] = data["final_decision"].map(y_map)
    plot_b_s_qs(data, ax, 3, "t2", 3)

    ax[0,0].set_title("Bowker test")
    ax[0,1].set_title("Stuart test")
    ax[0,2].set_title("QS test")
    ax[0,3].set_title("Procedure outcome")

    # draw line
    top_row_bottom = ax[1,0].get_position().y0
    bottom_row_top = ax[2,0].get_position().y1
    yline = (top_row_bottom + bottom_row_top) / 2
    line = Line2D([0.0, 1.0], [yline, yline],
                transform=fig.transFigure, color='black', lw=1)
    fig.add_artist(line)

    #fig.tight_layout(pad=1.5, w_pad=1.0, h_pad=1.0)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.8, hspace=1.0)
    plt.savefig(fig_dir / "figure1_raw.svg")
    plt.show()
    



