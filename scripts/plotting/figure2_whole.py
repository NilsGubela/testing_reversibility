import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from util import * 

ROOT = Path(__file__).resolve().parents[2]
out_dir = ROOT / "results" / "figure2"
data_dir = ROOT / "data"
fig_dir = ROOT / "figs"


meta = pd.read_csv(data_dir / "raw/sequences.csv")
meta["Collection_Date"] = pd.to_datetime(meta["Collection_Date"], errors="coerce")

cols = ["H_A_C", "H_C_A", "H_A_G", "H_G_A", "H_A_T", "H_T_A",
        "H_C_G", "H_G_C", "H_C_T", "H_T_C", "H_G_T", "H_T_G"]

subs_h = ["H_A_C", "H_C_A", "H_A_G", "H_G_A", "H_A_T", "H_T_A", "H_C_G", "H_G_C", "H_C_T", "H_T_C", "H_G_T", "H_T_G"]
labels_subs = ["AC", "CA", "AG", "GA", "AT", "TA", "CG", "GC", "CT", "TC", "GT", "TG"]

lower = ["2019-12-31","2020-12-31","2021-12-31","2022-12-31","2023-12-31","2024-12-31","2025-12-31"]
upper = ["2020-12-31","2021-12-31","2022-12-31","2023-12-31","2024-12-31","2025-12-31","2026-12-31"]

data_whole = pd.read_csv(out_dir / "all_seq_all_regions.csv")
data_whole = data_whole.rename(columns={"stat_b": "Bowker", "stat_s": "Stuart", "stat_qs": "QS"})
data_whole = evaluate_tests(data_whole)
data_whole = add_decision(data_whole)

data_whole = data_whole.merge(
    meta[["Accession", "Collection_Date"]],
    left_on="accession",
    right_on="Accession",
    how="left"
).drop(columns=["Accession"])

data_whole["t"] = data_whole["Collection_Date"]

data_whole["all_nonzero_flag"] = (
    data_whole[cols].ne(0).all(axis=1) & data_whole[cols].notna().all(axis=1)
).astype(int)
data_whole["real_final_decision"] = np.where(
    data_whole["all_nonzero_flag"].eq(1),
    data_whole["final_decision"],
    np.nan
)

data_whole["y"] = data_whole["real_final_decision"].map(y_map)


data_tmp = data_whole.dropna(subset=["t"])
data_tmp1 = data_tmp
data_tmp = data_tmp[data_tmp["y"] > 0]
with plt.rc_context(STYLE):
    fig = plt.figure(figsize=(double_col, 2*1.5))
    fig.suptitle("Whole SARS-CoV-2 genome", weight='bold', fontsize = 10)
    gs = fig.add_gridspec(2, 4, wspace=0.8, hspace=0.8)

    ax0 = [fig.add_subplot(gs[0, :2]),"placeholder",fig.add_subplot(gs[0, 2:])]
    ax = [fig.add_subplot(gs[1, i]) for i in range(4)]

    colors = {0: "red",1: "green",2: "green",3: "green",4: "green",}
    ax0[0].scatter(
        data_tmp1["Collection_Date"].values, data_tmp1["mut_sum"].values,
        c=[colors[g] for g in data_tmp1["y"]],
        s=5, alpha=0.85, edgecolors="none", zorder = 2
    )
    ax0[0].set_xlabel("Collection date")
    ax0[0].set_ylabel("Number of differences\nfrom reference")

    cmap = plt.get_cmap("PiYG")
    for i,sub_h in enumerate(subs_h):
        data_main = data_tmp1.loc[:,sub_h]/data_tmp1.loc[:,"mut_sum"]
        x = []
        y = []
        xi = 10*i
        for j in range(len(upper)):
            data = data_main[(data_tmp1["Collection_Date"] > lower[j]) & (data_tmp1["Collection_Date"] <= upper[j])]
            data = data.fillna(0)
            b = data.median()
            pos = xi+j*1.2
            y_errormin = [b - data.quantile(0.025)]
            y_errormax = [data.quantile(0.975) - b]
            c = [y_errormin, y_errormax]
            ax0[2].errorbar(pos, b, yerr=c, fmt='.', color=cmap(j/len(lower)),  capsize=2, capthick=0.5, ms = 0, elinewidth = 0.5)
            y.append(b)
            x.append(pos)
        ax0[2].plot(x,y, color = "black", linewidth = 1)
    ax0[2].set_xticklabels([])
    ax0[2].set_ylabel("Substitution frequency\n(normalised)")
    ax0[2].set_xticks(10*np.arange(0, len(subs_h))+4, labels_subs)
    ax0[2].set_xlabel("Substitution")

    # bottom

    ax[0].scatter(data_tmp["t"], data_tmp["Bowker"], s = 1, color = "steelblue")
    ax[0].axhline(chi2.ppf(0.95, 6), linestyle = "dashed", color = "k")
    ax[0].set_xlabel("Collection date")
    ax[0].set_ylabel(r"$S^2_B$", rotation=0)

    ax[1].scatter(data_tmp["t"], data_tmp["Stuart"], s = 1, color = "tab:purple")
    ax[1].axhline(chi2.ppf(0.95, 3), linestyle = "dashed", color = "k")
    ax[1].set_xlabel("Collection date")
    ax[1].set_ylabel(r"$S^2_S$", rotation=0)

    ax[2].scatter(data_tmp["t"], data_tmp["QS"], s = 1, color = "tab:orange")
    ax[2].axhline(norm.ppf(0.975), linestyle = "dashed", color = "k")
    ax[2].axhline(norm.ppf(0.025), linestyle = "dashed", color = "k")
    ax[2].set_xlabel("Collection date")
    ax[2].set_ylabel(r"$S^2_{QS}$", rotation = 0)

    values = [1,2,3,4]
    bc = np.bincount(data_tmp1["y"])
    counts = bc[values]
    colors = ["grey", "grey", "grey", "grey", "grey"]
    bars = ax[3].barh(values, np.round(counts/np.sum(counts) * 100,2), color = colors)
    ax[3].bar_label(bars, padding=0)
    ax[3].set_xlabel("Percentage")

    ax[3].set_yticks([1,2,3,4],labels[1:])

    for a in ax[:-1]:
        a.xaxis.set_major_locator(mdates.YearLocator(2))
        a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    ax0[0].xaxis.set_major_locator(mdates.YearLocator())
    ax0[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax[0].set_title("Bowker test")
    ax[1].set_title("Stuart test")
    ax[2].set_title("QS test")
    ax[3].set_title("Procedure outcome")

    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(fig_dir /"figure2_whole_raw.svg")
    plt.show()


