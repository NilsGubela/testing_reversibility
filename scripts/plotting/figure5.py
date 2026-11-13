import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tabulate import tabulate
from util import * 

ROOT = Path(__file__).resolve().parents[2]
fig_dir = ROOT / "figs"
out_dir = ROOT / "results" / "sequence_analysis"
data_dir = ROOT / "data" / "raw"

data_whole = pd.read_csv(out_dir/"all_orfs.csv")
meta = pd.read_csv(data_dir/"sequences.csv")
meta["Collection_Date"] = pd.to_datetime(meta["Collection_Date"], errors="coerce")
data_whole = data_whole.merge(
    meta[["Accession", "Collection_Date"]],
    left_on="accession",
    right_on="Accession",
    how="left"
).drop(columns=["Accession"])


row_titles = ["Whole SARS-CoV-2 genome", "SARS-CoV-2 spike region", "SARS-CoV-2 ORF 1a/b"]
colors = ["lightblue", "darkred", "grey"]
focus_regions = ["full", "S", "1a/b"] 

cols = ["H_A_C", "H_C_A", "H_A_G", "H_G_A", "H_A_T", "H_T_A",
        "H_C_G", "H_G_C", "H_C_T", "H_T_C", "H_G_T", "H_T_G"]

subs_h = ["H_A_C", "H_C_A", "H_A_G", "H_G_A", "H_A_T", "H_T_A", "H_C_G", "H_G_C", "H_C_T", "H_T_C", "H_G_T", "H_T_G"]
labels_subs = ["AC", "CA", "AG", "GA", "AT", "TA", "CG", "GC", "CT", "TC", "GT", "TG"]

lower = ["2019-12-31","2020-12-31","2021-12-31","2022-12-31","2023-12-31","2024-12-31","2025-12-31"]
upper = ["2020-12-31","2021-12-31","2022-12-31","2023-12-31","2024-12-31","2025-12-31","2026-12-31"]

orf_label = ["1a/b", "S", "rest", "full"]

with plt.rc_context(STYLE):

    fig, ax = plt.subplots(2,2,figsize=(double_col, 2*1.8),)

    n_bins = len(upper)
    x = np.arange(n_bins)

    # collect median values per region per time bin
    region_vals = {}

    for region in orf_label:
        if region == "full":
            continue
        data_tmp = data_whole[data_whole["orf"] == region]
        vals = []
        for j in range(n_bins):
            data = data_tmp[
                (data_tmp["Collection_Date"] > lower[j]) &
                (data_tmp["Collection_Date"] <= upper[j])
            ]["mut_sum"]
            
            vals.append(data.median() if len(data) > 0 else 0)
        region_vals[region] = vals

    # stacked bar plot
    bottom = np.zeros(n_bins)

    for i, region in enumerate(orf_label[:-1]):
        vals = np.array(region_vals[region])
        ax[0,0].bar(x, vals, bottom=bottom, color=colors[i], label=region, width=0.8)
        bottom += vals


    ax[0,0].set_xticks([0,1,2,3,4,5,6])    
    ax[0,0].set_xticklabels([2020, 2021, 2022, 2023, 2024, 2025, 2026])
    ax[0,0].set_ylabel("Median number of substitutions")
    ax[0,0].set_xlabel("Year")
    ax[0,0].set_ylim(top = 170)
    ax[0,0].legend( ncol = 2,loc="upper left", frameon=False)
    ax[0,0].set_title("Substitions per region")

    ax = ax.flatten()
    ax = ax[1:]
    colors = ["red", "tab:orange", "tab:purple", "steelblue", "grey"]

    for k, focus_region in enumerate(focus_regions):
        if focus_region == "all_but_S":
            data_tmp = data_whole[~data_whole["orf"].isin(["full", "S"])]
        else:
            data_tmp = data_whole[data_whole["orf"] == focus_region]

        cmap = plt.get_cmap("PiYG")
        for i,sub_h in enumerate(subs_h):
            data_main = data_tmp.loc[:,sub_h]/data_tmp.loc[:,"mut_sum"]
            x = []
            y = []
            xi = 10*i
            for j in range(len(upper)):
                data = data_main[(data_tmp["Collection_Date"] > lower[j]) & (data_tmp["Collection_Date"] <= upper[j])]
                data = data.fillna(0)
                b = data.median()
                pos = xi+j*1.2
                y_errormin = [b - data.quantile(0.025)]
                y_errormax = [data.quantile(0.975) - b]
                c = [y_errormin, y_errormax]
                ax[k].errorbar(pos, b, yerr=c, fmt='.', color=cmap(j/len(lower)),  capsize=2, capthick=0.5, ms = 0, elinewidth = 0.5)
                y.append(b)
                x.append(pos)
            ax[k].plot(x,y, color = "black", linewidth = 1)
        ax[k].set_xticklabels([])
        ax[k].set_ylabel("Substitution frequency\n(normalised)")
        ax[k].set_xticks(10*np.arange(0, len(subs_h))+4, labels_subs)
        ax[k].set_xlabel("Substitution")
        ax[k].set_title(row_titles[k])

    plt.tight_layout()
    plt.savefig(fig_dir / "figure5.pdf",dpi = 300, bbox_inches="tight" )