import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from util import * 

ROOT = Path(__file__).resolve().parents[2]
fig_dir = ROOT / "figs"
out_dir = ROOT / "results" / "simulation"

files = [
    out_dir/"result_mixture_stat_rev.csv",
    out_dir/"result_mixture_realistic_sitehet.csv",
    out_dir/"result_mixture_gtr.csv"]

eval_cols = [2,2,2]

alpha = 0.05

legend_elements = [
    Patch(facecolor="red", label="non-S/non-R"),
    Patch(facecolor="tab:orange",  label="non-S/R"),
    Patch(facecolor="tab:purple",  label="S/non-R"),
    Patch(facecolor="steelblue",  label="S/R"),
    Patch(facecolor="lightgrey",  label="non-S/NA"),
]


with plt.rc_context(STYLE):
    fig, ax = plt.subplots(
    1, 4,
    figsize=(double_col,  1.3),
    gridspec_kw={"width_ratios": [1, 1,1, 0.25]}
    )

    for i,file in enumerate(files):
        data = pd.read_csv(file)

        data = get_pvalue(data, "stat_b", "bowker_p", df = None, df_col = "bowker_df")
        data = get_pvalue(data, "stat_s", "stuart_p", df = 3)
        data = get_pvalue(data, "stat_qs", "qs_p", df = None, df_col = "qs_df")

        # evaluate testing procedure
        data = evaluate_test(data, "bowker_p", "bowker_decision")
        data = evaluate_test(data, "stuart_p", "stuart_decision")
        data = evaluate_test(data, "qs_p", "qs_decision")

        # total decision
        data["total_decision"] = np.select(
            [
                data["bowker_decision"] == 0,
                data["bowker_decision"].isna(),
                data["stuart_decision"] == 0,
                data["stuart_decision"].isna(),
                data["qs_decision"] == 0,
                data["qs_decision"].isna(),
            ],
            [4, 0, 3, 0, 2, 0,],
            default = 1
        )

        # bar plot per region
        orf_label2 = ["1a/b", "S", "rest", "full"]

        colors = ["lightgrey", "red", "tab:orange", "tab:purple", "steelblue"]
        
        for j, region in enumerate(orf_label2):
            tmp = data[(data["region"] == region)]
            counts = tmp["total_decision"].value_counts().reindex([0, 1, 2, 3,4], fill_value=0)
            bottom = 0
            for val, count in counts.items():
                ax[i].bar(j+0.5, count, bottom=bottom, label=str(val), color = colors[val])
                bottom += count

                if region == "full":
                    print(colors[val], count/500)

        ax[i].set_xticks(np.arange(4)+0.5)
        ax[i].set_xticklabels(orf_label2)
        ax[i].set_yticks([0, 250, 500])
        ax[i].set_yticklabels([0, 0.5, 1])
        ax[i].set_ylabel("Density")
        ax[i].set_xlabel("Region")
    
    ax[3].legend(handles=legend_elements,frameon = False)
    ax[3].axis('off')


    fig.tight_layout()
    plt.savefig(fig_dir / "figure3.pdf",dpi = 300, bbox_inches="tight" )