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
out_dir = ROOT / "results" / "simulation"

from scipy.linalg import expm


# theoretical part
pis = np.array([0.2, 0.3, 0.2, 0.3])
Q = np.array(   [[-(1-pis[0]), pis[1], pis[2], pis[3]],
                [pis[0], -(1-pis[1]), pis[2], pis[3]],
                [pis[0], pis[1], -(1-pis[2]), pis[3]],
                [pis[0], pis[1], pis[2], -(1-pis[3])]])
pi = np.array([0.1, 0.1, 0.4, 0.4])
ts = np.linspace(0.1,1,100)
res_b = []
res_s = []
n = 30000
for t in ts:
    P = expm(t * Q)
    H = np.matmul(np.diag(pi), P)
    stat, _ = bowker_stat2(n*H)
    res_b.append(stat)
    res_s.append(stuart_from_counts(n*H))

# Simulation part
data = pd.read_csv(out_dir/"result_ababneh_nonstationary_reversible_f81.csv")

data = get_pvalue(data, "stat_b", "bowker_p", df = None, df_col = "bowker_df")
data = get_pvalue(data, "stat_s", "stuart_p", df = 3)
data = get_pvalue(data, "stat_qs", "qs_p", df = None, df_col = "qs_df")

data["stat_ab"] = data["stat_b"] - data["stat_s"]
data["ab_df"] = data["bowker_df"] - 3
data = get_pvalue(data, "stat_ab", "ab_p", df = None, df_col = "ab_df")

alphas = np.arange(0,100)/100

line_qs = []
line_is = []
for alpha in alphas:
     line_qs.append(len(data[data["qs_p"]< alpha])/len(data))
     line_is.append(len(data[data["ab_p"]< alpha])/len(data))

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(1,2,figsize=(single_col, 1.5),)

    ax[0].plot(ts, np.array(res_b) - np.array(res_s), color = "tab:red")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel(r"$S^2_{IS}=S^2_B - S^2_S$")
     
    ax[1].plot(alphas, line_qs, label = "QS-test", color = "tab:orange")
    ax[1].plot(alphas, line_is, label = "IS-test", color = "tab:red")
    ax[1].plot(alphas, alphas, color = "black", linestyle = "dashed")
    ax[1].legend(frameon = False)
    ax[1].set_xlabel(r"$\alpha$")
    ax[1].set_ylabel("Rejection rate")
    fig.tight_layout()
    plt.savefig(fig_dir / "figureD1.pdf",dpi = 300, bbox_inches="tight" )