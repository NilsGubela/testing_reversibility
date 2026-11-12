import numpy as np
import sys
from pathlib import Path
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
    n = 30000
    k = 10000


    freq = np.array([[0.15],[0.1],[0.4],[0.35]])


    i = 0
    results = []

    master_seed = 12345

    #ts = np.arange(1,21)/20
    ts = [0.001, 0.00215, 0.005, 0.01, 0.0215, 0.05, 0.1, 0.215, 0.5]
    pbar = tqdm(total=k*len(ts), desc="Simulations", unit="rep")
    for t in ts:
        count = 0
        while count < k:
            # takeing care of seeds
            seed_seq = np.random.SeedSequence(
            [master_seed, i]
            )

            pi_seed, q_seed, seq_seed, mut_seed = seed_seq.generate_state(4)

            # sample pi for Q
            pi = rate_matrix_cpp.get_pi(pi_seed)

            # sample random reversible Q with init dist pi
            Q = rate_matrix_cpp.get_Q_from_pi(pi, q_seed)

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
                "stat_qs": stat_qs,
                "bowker_df": df_b,
                "stat_qs": stat_qs,
                "qs_df": df_qs,
                'db_deviation': return_detailed_balance(Q, pi),
            })

            count += 1
            pbar.update(1)

    pbar.close()

    script_dir = Path(__file__).resolve().parent         
    out_dir = ROOT2 / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "error_analysis/result_type1.csv"

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)
    print(f"Saved {len(results_df)} sims to: {out_path}")


if __name__ == "__main__":
    main()