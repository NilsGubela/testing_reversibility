import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--vary",
    required=True,
    choices=["divergence", "sequence_length", "db_deviation"]
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


    freq = np.array([[0.15],[0.1],[0.4],[0.35]])
    t = 0.1

    # standard values
    seq_lengths = [30000]
    ts = [0.5]

    out_dir = ROOT2 / "results"
    #out_dir.mkdir(parents=True, exist_ok=True)

    # full values
    if args.vary == "divergence":
        #ts = np.arange(1,21)/20
        ts = [0.001, 0.00215, 0.005, 0.01, 0.0215, 0.05, 0.1, 0.215, 0.5]
        out_path = out_dir / "error_analysis/result_type2_divergence.csv"
    elif args.vary == "sequence_length":
        seq_lengths = np.unique(np.logspace(np.log10(100), np.log10(50000), num=10).astype(int))
        out_path = out_dir / "error_analysis/result_type2_length.csv"
    elif args.vary == "db_deviation":
        out_path = out_dir / "error_analysis/result_type2_db_deviation.csv"
    
    i = 0
    results = []

    master_seed = 12345

    pbar = tqdm(total=k * len(ts) * len(seq_lengths), desc="Simulations", unit="rep")

    for t in ts:
        for n in seq_lengths:

            count = 0

            while count < k:
                # takeing care of seeds
                seed_seq = np.random.SeedSequence(
                [master_seed, i]
                )

                q_seed, seq_seed, mut_seed = seed_seq.generate_state(3)

                # sample random nonreversible Q with init dist pi
                Q = rate_matrix_cpp.get_Q_nonrev(q_seed)
                vals, vecs = np.linalg.eig(Q.T)
                pi = np.real(vecs[:, np.argmin(np.abs(vals))])
                pi = pi / pi.sum()   # normalize
                # accept with same probability:
                if return_detailed_balance(Q, pi) < np.random.rand():
                    # sampel again
                    i += 1
                    continue


                # sample ancestral sequence
                seq0 = rand_seq_cpp.rand_seq(n,freq,int(seq_seed))

                # calculate P
                P = mat_exp_cpp.mat_exp_4d(Q, t)

                # simulate evolution
                seq1 = seq_sim_cpp.seq_sim(seq0,P,int(mut_seed))

                # calculate sample diversity matrix
                H = div_mat_cpp.div_mat(seq0, seq1)

                # calculate statistics 
                stat_b, df_b = bowker_stat2(H)
                stat_s = stuart_from_counts(H)

                stat_qs, df_qs = qs_test_with_df(H,n)
                i += 1
                results.append({
                    "t": t,
                    "n": n,
                    "mut_sum": H.sum() - np.trace(H),
                    "stat_b": stat_b,
                    "stat_s": stat_s,
                    "bowker_df": df_b,
                    "stat_qs": stat_qs,
                    "qs_df": df_qs,
                    'db_deviation': return_detailed_balance(Q, pi),
                })

                count += 1
                pbar.update(1)
                pbar.set_postfix(rejects=i - count)

    pbar.close()

    script_dir = Path(__file__).resolve().parent         
    out_dir = ROOT2 / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)
    print(f"Saved {len(results_df)} sims to: {out_path}")


if __name__ == "__main__":
    main()