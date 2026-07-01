import numpy as np
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

y_map = {name: i for i, name in enumerate(order)}

