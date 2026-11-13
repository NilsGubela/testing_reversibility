import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import norm

STYLE = {
    "font.family": "sans-serif",
    "font.size": 6,
    "axes.labelsize": 8,
    "axes.titlesize": 10,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.sans-serif": ["Times"],
    'svg.fonttype': 'none',
}

STYLE2 = {
    "font.family": "sans-serif",
    "font.size": 6,
    "axes.labelsize": 8,
    "axes.titlesize": 10,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.sans-serif": ["Times"],
    'svg.fonttype': 'path',
}

single_col = 3.42
single_half_col = 4.5
double_col = 6.8
max_height = 9

def evaluate_tests(data, alpha = 0.05):
    # Bowker test
    df = 6
    crit = chi2.ppf(1 - alpha, df)
    x = data["Bowker"]
    data["b_decision"] = np.select(
        [x.isna(), x > crit],
        ["reject", "reject"],
        default="accept"
    )
    data["b_pvalue"] = 1 - chi2.cdf(x, df)

    # Stuart test
    df = 3
    crit = chi2.ppf(1 - alpha, df)
    x = data["Stuart"]
    data["s_decision"] = np.select(
        [x.isna(), x > crit],
        ["reject", "reject"],
        default="accept"
    )
    data["s_pvalue"] = 1 - chi2.cdf(x, df)

    # qs test
    zcrit = norm.ppf(1 - alpha/2)
    z = data["QS"]
    data["qs_decision"] = np.select(
        [
            z.isna(),
            z.abs() > zcrit
        ],
        [
            "reject",
            "reject"
        ],
        default="accept"
    )
    data["qs_p_value"] = 2 * (1 - norm.cdf(data["QS"].abs()))

    return(data)


def add_decision(df):
    df["final_decision"] = np.select(
    [
        df["b_decision"].eq("accept"),
        df["s_decision"].eq("accept"),
        df["qs_decision"].eq("accept"),
    ],
    [
        "stationary reversible",
        "stationary nonreversible",
        "nonstationary reversible",
    ],
    default="nonstationary nonreversible",
    )
    return(df)


order = [
    np.nan,
    "nonstationary nonreversible",
    "nonstationary reversible",
    "stationary nonreversible",
    "stationary reversible",
]

labels = [
    "NA",
    "non-S\n+non-R",
    "non-S+R",
    "S+non-R",
    "S+R",
]


def get_pvalue(data, stat_col, res_col, df = 3, df_col = None):
    if df is None:
        # use df_col for degrees of freedom
        #data[res_col] = chi2.sf(data[stat_col], df = data[df_col])
        data[res_col] = np.where(
            data[df_col] == 0,
            np.nan,
            chi2.sf(data[stat_col], df=data[df_col])
        )
        
    else:
        data[res_col] = chi2.sf(data[stat_col], df = df)
    return(data)

def evaluate_test(data, p_col, res_col, alpha = 0.05):
    data[res_col] = np.where(
        data[p_col].isna(),
        np.nan,
        np.where(data[p_col] <= alpha, 1, 0)
    )
    return(data)


def evaluate_qs_qf(df, alpha = 0.05, stat_col = "stat_qs", df_col = "qs_df", res_col = "qs_qf_decision"):
    df[res_col] = np.where(
        df[stat_col].isna(),
        np.nan,
        np.where(
            df[stat_col] <= chi2.ppf(1 - alpha, df[df_col]),
            0,
            1
        )
    )
    return(df)

def plot_strategy(data, ax, row, t, col_index, test_cols = ["bowker_p", "stuart_p", "qs_p", "ab_p"], alpha = 0.05):
    colors = ["steelblue", "tab:purple", "tab:orange", "red", "forestgreen"]
    labels = [r"$-\log_{10}(p)$", r"$-\log_{10}(p)$", r"$-\log_{10}(p)$", r"$-\log_{10}(p)$"]
    eval_cols = ["grey","grey","grey","grey","grey"]
    eval_cols[col_index] = "forestgreen"
    for i, pcol in enumerate(test_cols):
        p = np.clip(data[pcol].values, 1e-10, 1.0)
        y = - np.log10(p)
        ax[row,i].scatter(data[t], y, s = 0.1, color = colors[i])
        ax[row,i].axhline(-np.log10(alpha), linestyle = "dashed", color = "k")
        ax[row,i].set_xlabel(r"$t$")
        ax[row,i].set_ylabel(labels[i], fontsize = 6)

        #print(labels[i], " rejection: ", len(data[data[pcol]< alpha])/len(data))
    if t == "t2":
        ax[row,0].set_xlabel(r"$t_2$")
        ax[row,1].set_xlabel(r"$t_2$")
        ax[row,2].set_xlabel(r"$t_2$")
        ax[row,3].set_xlabel(r"$t_2$")

    last_index = len(test_cols)
    values = [0,1,2,3,4]
    bc = np.bincount(data["total_decision"])
    counts = bc[values]
    bars = ax[row,last_index].barh(values, counts/np.sum(counts) * 100, color = eval_cols)
    ax[row,last_index].bar_label(bars, padding=0)
    ax[row,last_index].set_xlabel("Percentage")
    ax[row,last_index].set_yticks([0,1,2,3,4],["NA","non-S\n+non-R","non-S+R","S+non-R","S+R"])


def bh_correct(df: pd.DataFrame, p_col: str, out_col: str, alpha: float = 0.05) -> pd.DataFrame:
    out = df.copy()
    p = out[p_col].to_numpy(dtype=float)

    n = len(p)
    p_adj = np.full(n, np.nan)

    valid = ~np.isnan(p)
    p_valid = p[valid]
    m = len(p_valid)

    if m > 0:
        order = np.argsort(p_valid)
        ranked_p = p_valid[order]

        adj_sorted = ranked_p * m / (np.arange(1, m + 1))
        adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
        adj_sorted = np.clip(adj_sorted, 0, 1)

        p_adj_valid = np.empty(m)
        p_adj_valid[order] = adj_sorted
        p_adj[valid] = p_adj_valid

    out[p_col + "_adj"] = p_adj
    #out[out_col] = 1*(out[p_col + "_adj"] <= alpha)
    out[out_col] = np.where(np.isnan(out[p_col + "_adj"]), np.nan, (out[p_col + "_adj"] <= alpha).astype(float))

    return out

def browns_method(pvals):
    pvals = np.asarray(pvals, dtype=float)
    pvals = pvals[~np.isnan(pvals)]
    n = len(pvals)

    if n == 0:
        return {"n_obs": 0, "brown_stat": np.nan, "p_value": np.nan}
    if n == 1:
        stat = -2 * np.log(pvals[0])
        return {"n_obs": 1, "brown_stat": stat, "p_value": pvals[0]}

    stat = -2 * np.sum(np.log(pvals))

    expected = 2 * n
    var = 4 * n  # equals Fisher under independence
    c = var / (2 * expected)
    df = 2 * expected**2 / var

    p_comb = 1 - chi2.cdf(stat / c, df)

    return {"n_obs": n, "brown_stat": stat, "p_value": p_comb}

def bowker_stat2(M):
    M = np.asarray(M, dtype=float)

    if M.shape != (4, 4):
        raise ValueError("Input must be a 4x4 matrix.")

    stat = 0.0
    df = 0

    for i in range(4):
        for j in range(i + 1, 4):
            a = M[i, j]
            b = M[j, i]
            s = a + b
            if s > 0:
                stat += (a - b) ** 2 / s
                df += 1

    return stat, df

def stuart_from_counts(N):
    N = np.asarray(N)
    if N.ndim != 2 or N.shape[0] != N.shape[1]:
        raise ValueError("counts must be a square 2D array.")
    K = N.shape[0]
    if K < 2:
        return 0.0 # stat, df, rank
    r = N.sum(axis=1)
    c = N.sum(axis=0)
    d_full = (r - c).astype(float)

    m = K - 1
    d = d_full[:m]

    V = np.zeros((m, m), dtype=float)
    for i in range(m):
        V[i, i] = r[i] + c[i] - 2 * N[i, i]
        for j in range(m):
            if i != j:
                V[i, j] = -(N[i, j] + N[j, i])

    rank = int(np.linalg.matrix_rank(V))
    if rank == 0:
        return np.nan
    stat = float(d.T @ np.linalg.pinv(V) @ d)
    return stat

