import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D
from util import * 

ROOT = Path(__file__).resolve().parents[2]
fig_dir = ROOT / "figs"
out_dir = ROOT / "results" / "simulation"


files = [out_dir/"result_nonstationary_reversible_cherry.csv",
         out_dir/"result_stationary_nonreversible_cherry.csv"]

eval_cols = [2,3]

alpha = 0.05

with plt.rc_context(STYLE):
    #fig, ax = plt.subplots(4,5,figsize=(6.29, 4*1.3),)
    fig, ax = plt.subplots(
    2, 4,
    figsize=(double_col, 2 * 1.5),
    gridspec_kw={"width_ratios": [1, 1, 1, 0.45]}
)

    for i,file in enumerate(files):
        data = pd.read_csv(file)
        #data = compute_qs_qf(data)
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

    
        plot_strategy(data, ax, i, "t2", eval_cols[i], test_cols = ["bowker_p", "stuart_p", "qs_p"], alpha = 0.05)
        
    ax[0,0].set_title("Bowker test")
    ax[0,1].set_title("Stuart test")
    ax[0,2].set_title("QS test")
    #ax[0,3].set_title("IS test")
    ax[0,3].set_title("Procedure outcome")


    ref = ax[0, 0]

    for a in ax[:, :-1].flat:
        a.set_ylim(ref.get_ylim())

    fig.tight_layout()
    plt.savefig(fig_dir / "figure2.pdf",dpi = 300, bbox_inches="tight" )