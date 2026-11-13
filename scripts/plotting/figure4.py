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


legend_elements = [
    Patch(facecolor="red", label="non-S/non-R"),
    Patch(facecolor="tab:orange",  label="non-S/R"),
    Patch(facecolor="tab:purple",  label="S/non-R"),
    Patch(facecolor="steelblue",  label="S/R"),
    Patch(facecolor="lightgrey",  label="non-S/NA"),
]

# BH correction
meta = pd.read_csv(data_dir/"sequences.csv")
meta["Collection_Date"] = pd.to_datetime(meta["Collection_Date"], errors="coerce")

cols = ["bowker-p", "stuart-p", "qs_qf-p"]

orf_label = ["1a/b", "S", "rest", "full"]

data_whole = pd.read_csv(out_dir/"all_orfs.csv")
#data_whole = compute_qs_qf(data_whole)

data_whole["n_used%"] = data_whole["n_used"]/data_whole["n"]

alpha = 0.05
# calculate Bowker p-value
data_whole["bowker-p"] = chi2.sf(data_whole["stat_b"],data_whole["bowker_df"])

# calculate Stuart p-value
data_whole["stuart-p"] = chi2.sf(data_whole["stat_s"],3)

# calculate qs qf p-value
data_whole["qs_qf-p"] = chi2.sf(data_whole["stat_qs"],data_whole["qs_df"])

rows = []
for (orf, lineage), g in data_whole.groupby(["orf", "pangolin_lineage"]):
    row = {"orf": orf, "pangolin_lineage": lineage}
    for col in cols:
        res = browns_method(g[col])
        row[f"{col}_n_obs"] = res["n_obs"]
        row[f"{col}_brown_stat"] = res["brown_stat"]
        row[f"{col}_p_value"] = res["p_value"]
        row["mut_sum"] = np.mean(g["mut_sum"])
    rows.append(row)

brown_table = pd.DataFrame(rows)

brown_table = (
    brown_table
    .groupby("orf", group_keys=False)
    .apply(lambda g: bh_correct(g, p_col="bowker-p_p_value", out_col = "bowker_decision", alpha=0.05))
)

# remove stuart and qs_qf p-values 
brown_table.loc[brown_table["bowker_decision"] == 0,"stuart-p_p_value"] = np.nan
brown_table.loc[brown_table["bowker_decision"] == 0,"qs_qf-p_p_value"] = np.nan
brown_table = (
    brown_table
    .groupby("orf", group_keys=False)
    .apply(lambda g: bh_correct(g, p_col="stuart-p_p_value", out_col = "stuart_decision", alpha=0.05))
)

# remove stuart and qs_qf p-values 
brown_table.loc[brown_table["stuart_decision"] == 0,"qs_qf-p_p_value"] = np.nan
brown_table = (
    brown_table
    .groupby("orf", group_keys=False)
    .apply(lambda g: bh_correct(g, p_col="qs_qf-p_p_value", out_col = "qs_qf_decision", alpha=0.05))
)

# total decision
brown_table["total_decision"] = np.select(
    [
        brown_table["bowker_decision"] == 0,
        brown_table["stuart_decision"] == 0,
        brown_table["qs_qf_decision"].isna(),
        brown_table["qs_qf_decision"] == 0
    ],
    [4, 3, 5, 2],
    default=1
)

rows = []

for region in orf_label:
    data_tmp = brown_table[brown_table["orf"] == region]
    data_tmp2 = data_whole[data_whole["orf"] == region]
    rows.append([
        region,
        len(data_tmp),
        data_tmp["total_decision"].isna().sum(),
        (data_tmp["total_decision"] == 1).sum(),
        (data_tmp["total_decision"] == 2).sum(),
        (data_tmp["total_decision"] == 3).sum(),
        (data_tmp["total_decision"] == 4).sum(),
        (data_tmp["total_decision"] == 5).sum(),
        (data_tmp2["n_used%"]).mean(),
    ])

print(tabulate(rows, headers=["Region", "# Lineages", "No decision", "N/N", "N/R", "S/N", "S/R", "No Symmetry, qs NaN", "Mean completeness"], tablefmt="grid"))

meta["Collection_Date"] = pd.to_datetime(meta["Collection_Date"], errors="coerce")

earliest_dates = (
    meta.groupby("Pangolin", as_index=False)["Collection_Date"]
    .min()
    .rename(columns={"Pangolin": "pangolin_lineage"})
)

earliest_dates = earliest_dates.sort_values("Collection_Date").reset_index(drop = True).reset_index()
earliest_dates["Collection_Date"] = pd.to_datetime(earliest_dates["Collection_Date"])
earliest_dates["year"] = earliest_dates["Collection_Date"].dt.year

brown_table = brown_table.merge(earliest_dates, on="pangolin_lineage", how="left")
data_whole = data_whole.merge(
    meta[["Accession", "Collection_Date"]],
    left_on="accession",
    right_on="Accession",
    how="left"
).drop(columns=["Accession"])

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(
    1, 3,
    figsize=(double_col,  2.5),
    gridspec_kw={"width_ratios": [1, 0.25,1]}
    )
    colors = ["red", "tab:orange", "tab:purple", "steelblue", "lightgrey"]
    for lineage in brown_table["pangolin_lineage"].unique():
        for i, region in enumerate(orf_label):
            tmp = brown_table[(brown_table["orf"] == region)&(brown_table["pangolin_lineage"] == lineage)]
            if len(tmp) == 0 or lineage == "B.1.329":
                continue
            index = tmp["index"].values[0]
            index_color = tmp["total_decision"].values[0] - 1
            ax[0].fill_between([i, i+1], index, [index + 1, index + 1], color = colors[index_color])


    year_starts = earliest_dates.groupby("year", sort=False).apply(lambda x: x.index[0])

    ax[0].set_yticks(year_starts.values[::2])
    ax[0].set_yticklabels([2020, 2022, 2024, 2026])
    ax[0].set_yticks(year_starts.values[1::2], minor=True) 
    ax[0].set_xticks(np.arange(4)+0.5)
    ax[0].set_xticklabels(orf_label)
    ax[0].set_xlabel("Region")
    ax[0].set_ylabel("Collection Date")

    ax[1].legend(handles=legend_elements,frameon = False)
    ax[1].axis('off')

    ax[0].set_xmargin(0)
    ax[0].set_ymargin(0)

    # procedure outcome
    colors = ["red", "tab:orange", "tab:purple", "steelblue","lightgrey"]
    for j, region in enumerate(orf_label):
        tmp = brown_table[(brown_table["orf"] == region)]
        counts = tmp["total_decision"].value_counts().reindex([5, 1, 2, 3,4], fill_value=0)
        bottom = 0
        for val, count in counts.items():
            ax[2].bar(j+0.5, count/len(tmp), bottom=bottom, label=str(val), color = colors[val-1])
            bottom += count/len(tmp)
    ax[2].set_xticks(np.arange(4)+0.5)
    ax[2].set_xticklabels(orf_label)
    ax[2].set_ylabel("Density")
    ax[2].set_xlabel("Region")

    plt.savefig(fig_dir / "figure4.pdf",dpi = 300, bbox_inches="tight" )