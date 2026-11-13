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
out_dir2 = ROOT / "results" / "error_analysis"





with plt.rc_context(STYLE2):
    files = [out_dir/"result_nonstationary_reversible.csv",
         out_dir/"result_stationary_nonreversible.csv",
         ]

eval_cols = [2,3]

alpha = 0.05

with plt.rc_context(STYLE):
    fig = plt.figure(figsize=(double_col, 4 * 1.5))
    gs = GridSpec(4, 4, figure=fig, width_ratios=[1, 1, 1, 0.5])

    # Row 0
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax03 = fig.add_subplot(gs[0, 3])

    # Row 1: two equally sized plots
    gs_row1 = gs[1, :].subgridspec(1, 5, width_ratios=[0.26, 1,0.15, 1, 0.26], wspace=0.3)
    ax10 = fig.add_subplot(gs_row1[0, 1])
    ax11 = fig.add_subplot(gs_row1[0, 3])

    # Row 2
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])
    ax22 = fig.add_subplot(gs[2, 2])
    ax23 = fig.add_subplot(gs[2, 3])

    # Row 3: four equally sized plots
    gs_row3 = gs[3, :].subgridspec(1, 5, width_ratios=[0.27, 1,0.12, 1, 0.27], wspace=0.3)
    ax30 = fig.add_subplot(gs_row3[0, 1])
    ax31 = fig.add_subplot(gs_row3[0, 3])


    ax = np.array([[ax00, ax01, ax02, ax03], [ax20, ax21, ax22,ax23]])
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

        plot_strategy(data, ax, i, "t", eval_cols[i], test_cols = ["bowker_p", "stuart_p", "qs_p"], alpha = 0.05)
        ref = ax[0, 0]
        for a in ax[i, :-1]:
            a.set_ylim(ref.get_ylim())
    ax[0,0].set_title("Bowker test")
    ax[0,1].set_title("Stuart test")
    ax[0,2].set_title("QS test")
    ax[0,3].set_title("Procedure outcome")

    ax[1,0].set_title("Bowker test")
    ax[1,1].set_title("Stuart test")
    ax[1,2].set_title("QS test")
    ax[1,3].set_title("Procedure outcome")
    # draw line
    yline = 0.5#ax[1,0].get_position().y1
    line = Line2D([0.0, 1.0], [yline, yline],
                transform=fig.transFigure, color='black', lw=1)
    fig.add_artist(line)

    # type 1 error
    alphas = np.arange(1,100)/100

    data = pd.read_csv(out_dir2/"result_type1.csv")
    data = evaluate_qs_qf(data)

    data["mut_range"] = np.round(np.log10(data["mut_sum"]),1)
    ts = data["t"].unique()
    mut_range = data["mut_range"].unique()
    mut_range.sort()
    colors = ["darkblue", "forestgreen", "darkred"]
    for i,a in enumerate([0.05, 0.1, 0.5]):
        data = evaluate_qs_qf(data, alpha = a)
        x = []
        y = []
        for ms in mut_range:
            tmp = data[(data["mut_range"] == ms)]
            if len(tmp) > 100:
                ax11.scatter(ms, np.nanmean(tmp["qs_qf_decision"]), color = colors[i], s = 5)
                x.append(ms)
                y.append(np.nanmean(tmp["qs_qf_decision"]))
        ax11.plot(x,y, color = colors[i], label = rf"$\alpha = {a}$")
    ax11.set_xlabel("#Substitutions")
    ax11.set_ylabel(r"Type-1 error")
    ax11.legend(frameon= False)
    ax11.set_xticks([1,2,3,4])
    ax11.set_xticklabels([10,100,1000,10000])

    data = data[(data["t"] == 0.5) & (data["qs_df"] == 3)]
    data["stat_ab"] = data["stat_b"] - data["stat_s"]   
    data["ab_df"] = data["bowker_df"] - 3
    data = get_pvalue(data, "stat_ab", "ab_p", df = None, df_col = "ab_df")
    type1 = np.zeros_like(alphas)
    type1_ab = np.zeros_like(alphas)
    for i,alpha in enumerate(alphas):
        data = evaluate_qs_qf(data, alpha = alpha)
        type1[i] = np.nanmean(data["qs_qf_decision"])
        type1_ab[i] = len(data[data["ab_p"]< alpha])/len(data)
    ax10.plot(alphas, type1, color = 'tab:orange', label = "QS test")#
    ax10.plot(alphas, type1_ab, color = 'red', label = "IS test")
    ax10.plot(alphas, alphas, color = "grey", linestyle = "dashed")
    ax10.set_xlabel(r"$\alpha$")
    ax10.set_ylabel("Type-1-error")
    ax10.legend(frameon=False)
    fig.text(0.5, 0.73, "Type 1 error analysis", ha="center", va="top", size = 10)

    # statistical power
    data = pd.read_csv(out_dir2/"result_type2_divergence.csv")
    data = evaluate_qs_qf(data)

    data["mut_range"] = np.round(np.log10(data["mut_sum"]),1)
    muts = []
    line = []
    counts = []
    tmp_list = data["mut_range"].unique()
    tmp_list.sort()

    for i,a in enumerate([0.05, 0.1, 0.5]):
        data = evaluate_qs_qf(data, alpha = a)
        x = []
        y = []
        for ms in mut_range:
            tmp = data[(data["mut_range"] == ms)]
            if len(tmp) > 100:
                ax31.scatter(ms, np.sum(tmp["qs_qf_decision"])/len(tmp["qs_qf_decision"]), color = colors[i], s = 5)
                x.append(ms)
                y.append(np.sum(tmp["qs_qf_decision"])/len(tmp["qs_qf_decision"]))
        ax31.plot(x,y, color = colors[i], label = rf"$\alpha = {a}$")
    
    ax31.set_xlabel("#Substitutions")
    ax31.set_ylabel("Power")
    ax31.set_xticks([1,2,3,4])
    ax31.set_xticklabels([10,100,1000,10000])
    ax31.legend(frameon= False)



    data = pd.read_csv(out_dir2/"result_type2_db_deviation.csv")
    data = evaluate_qs_qf(data)
    data["bin"] = np.floor(data["db_deviation"]*20)/20
    bins = data["bin"].unique()
    bins.sort()
    line = []
    for bin in bins:
        tmp = data[data["bin"]==bin]
        ax30.scatter(bin+0.05, np.nanmean(tmp["qs_qf_decision"]), color = "k", s = 5)
        line.append(np.nanmean(tmp["qs_qf_decision"]))
    ax30.plot(bins + 0.05, line, color = "k")
    ax30.set_xlabel("Deviation from detailed balance")
    ax30.set_ylabel("Power")


    fig.text(0.5, 0.25, "Statistical power analysis", ha="center", va="top", size = 10)
    fig.tight_layout()
    plt.savefig(fig_dir / "figure1.pdf",dpi = 300, bbox_inches="tight" )
