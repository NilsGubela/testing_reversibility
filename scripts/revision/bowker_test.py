import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--H",
    required=True,
    choices=["H0", "H1"]
)

args = parser.parse_args()


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
    k = 10000


    #freq = np.array([[0.15],[0.1],[0.4],[0.35]])
    #t = 0.1

    # standard values
    seq_lengths = [3000, 30000]
    #ts = [0.5]
    ts = [0.001, 0.00215, 0.005, 0.01, 0.0215, 0.05, 0.1, 0.215, 0.5]

    out_dir = ROOT2 / "results"
    #out_dir.mkdir(parents=True, exist_ok=True)

    # full values
    if args.H == "H0":
        #ts = np.arange(1,21)/20
        #ts = [0.001, 0.00215, 0.005, 0.01, 0.0215, 0.05, 0.1, 0.215, 0.5]
        out_path = out_dir / "revision/result_bowker_test_h0.csv"
    elif args.H == "H1":
        #seq_lengths = np.unique(np.logspace(np.log10(100), np.log10(50000), num=10).astype(int))
        out_path = out_dir / "revision/result_bowker_test_h1.csv"
    
    i = 0
    
    master_seed = 12345

    pbar = tqdm(total=k * len(ts) * len(seq_lengths), desc="Simulations", unit="rep")

    results = []

    for t in ts:
        for n in seq_lengths:

            count = 0

            while count < k:
                # takeing care of seeds
                seed_seq = np.random.SeedSequence(
                [master_seed, i]
                )

                q_seed, pi_seed, seq_seed, mut_seed = seed_seq.generate_state(4)

                # sample pi for Q
                pi = rate_matrix_cpp.get_pi(pi_seed)

                # sample random nonreversible Q with init dist pi
                Q = rate_matrix_cpp.get_Q_from_pi(pi, q_seed)

                # sample ancestral sequence
                if args.H == "H0":
                    freq = pi
                elif args.H == "H1":
                    freq = np.array([[0.15],[0.1],[0.4],[0.35]])

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
                

                # generate diversity matrix with reference 
                stat_s = stuart_from_counts(H)
                stat_qs = qs_test(H, n, 3)
                stat_b, df_b = bowker_stat2(H)
                P_hat = H / n
                d = get_d(H)
                d1 = d[0] / np.sqrt(get_var(P_hat, n, 1))
                d2 = d[1] / np.sqrt(get_var(P_hat, n, 2))
                d3 = d[2] / np.sqrt(get_var(P_hat, n, 3))
                d4 = d[3] / np.sqrt(get_var(P_hat, n, 4))

                results.append({
                    "t": t,
                    "n": n,
                    "mut_sum": H.sum() - np.trace(H),
                    "stat_b": stat_b,
                    "stat_s": stat_s,
                    "stat_qs": stat_qs,
                    "bowker_df": df_b,
                    "d1": d1,
                    "d2": d2,
                    "d3": d3,
                    "d4": d4
                })
                count += 1
                i += 1
                pbar.update(1)

    pbar.close()

    script_dir = Path(__file__).resolve().parent         
    out_dir = ROOT2 / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    #out_path = out_dir / "error_analysis/result_type2.csv"

    # Save as one-column CSV
    #pd.DataFrame({"t": res_t, "Bowker": res_b, "Stuart": res_s, "QS": res_qs, "d1": res_d1, "d2": res_d2, "d3": res_d3, "d4": res_d4, "mut_sum": res_mut, "db_deviation": res_db_deviation, "n": res_n}).to_csv(out_path, index=False)
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)

    print(f"Saved {len(results_df)} sims to: {out_path}")


if __name__ == "__main__":
    main()