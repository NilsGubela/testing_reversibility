import numpy as np
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROOT2 = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT))
from util import *

sys.path.append(str(ROOT / "build"))

import rand_seq_cpp
import seq_sim_cpp
import mat_exp_cpp
import div_mat_cpp
import rate_matrix_cpp


def main():
    parser = argparse.ArgumentParser(description="Simulation study for d statistic.")
    parser.add_argument("--n", type=int, required=True, help="sequence length")
    parser.add_argument("--k", type=int, required=True, help="number of valid replicates to collect")
    parser.add_argument("--t", type=float, required=True, help="evolution time")
    args = parser.parse_args()

    n = args.n
    k = args.k
    t = args.t

    freq = np.array([[0.25],[0.25],[0.25],[0.25]])

    count = 0
    i = 0
    res_qs = []
    res_mut = []
    res_d1 = []
    res_d2 = []
    res_d3 = []
    res_d4 = []

    master_seed = 12345

    pbar = tqdm(total=k, desc="Simulations", unit="rep")

    while count < k:
        # takeing care of seeds
        seed_seq = np.random.SeedSequence(
        [master_seed, i]
        )

        q_seed, seq_seed, mut_seed = seed_seq.generate_state(3)

        # sample random reversible Q
        Q = rate_matrix_cpp.get_Q(q_seed)

        # sample ancestral sequence
        seq0 = rand_seq_cpp.rand_seq(n,freq,int(seq_seed))

        # calculate P
        P = mat_exp_cpp.mat_exp_4d(Q, t)

        # simulate evolution
        seq1 = seq_sim_cpp.seq_sim(seq0,P,int(mut_seed))

        # calculate sample diversity matrix
        H = div_mat_cpp.div_mat(seq0, seq1)

        # check if matrix has zero non-diagonal entry
        mat_no_diag = H.copy()
        np.fill_diagonal(mat_no_diag, np.nan)
        if np.any(mat_no_diag == 0):
            # sample again
            i += 1
            continue
        
        P_hat = H / n
        d = get_d(H)
        d1 = d[0] / np.sqrt(get_var(P_hat, n, 1))
        d2 = d[1] / np.sqrt(get_var(P_hat, n, 2))
        d3 = d[2] / np.sqrt(get_var(P_hat, n, 3))
        d4 = d[3] / np.sqrt(get_var(P_hat, n, 4))
        stat_qs = qs_test(H, n, 3)

        i += 1
        res_qs.append(stat_qs)
        res_d1.append(d1)
        res_d2.append(d2)
        res_d3.append(d3)
        res_d4.append(d4)
        np.fill_diagonal(mat_no_diag, 0)
        res_mut.append(np.sum(mat_no_diag))
        count += 1
        pbar.update(1)
        pbar.set_postfix(rejects=i - count)


    pbar.close()

    script_dir = Path(__file__).resolve().parent          
    out_dir = ROOT2 / "results" / "figure3"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_str = str(t).replace(".", "p")
    out_path = out_dir / f"result_{k}_{n}_{t_str}.csv"

    # Save as one-column CSV
    #pd.DataFrame({"stat": res}).to_csv(out_path, index=False)
    pd.DataFrame({"QS": res_qs, "d1": res_d1, "d2": res_d2, "d3": res_d3, "d4": res_d4, "mut_sum": res_mut}).to_csv(out_path, index=False)

    print(f"Saved {len(res_qs)} stats to: {out_path}")


if __name__ == "__main__":
    main()