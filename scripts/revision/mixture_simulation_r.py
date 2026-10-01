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
    n = 29903
    k = 10000

    freq = np.array([[0.3],[0.2],[0.2],[0.3]])

    orf_lens = [13202, 8086, 3822, 828, 228, 669, 186, 336, 336, 1260, 117]
    orf_start = np.cumsum(orf_lens) - orf_lens
    orf_end = np.cumsum(orf_lens)

    i = 0
    results = []

    master_seed = 12345

    #ts = [0.01, 0.0215, 0.05, 0.1, 0.215, 0.5]
    ts = [0.001, 0.00215, 0.005, 0.01, 0.0215, 0.05, 0.075]
    alphas = [0.1, 0.5, 5]
    pbar = tqdm(total=k*len(ts)*len(alphas), desc="Simulations", unit="rep")
    for t in ts:
        for alpha in alphas:
            count = 0
            while count < k:
                # takeing care of seeds
                seed_seq = np.random.SeedSequence(
                [master_seed, i]
                )

                pi_seed, q_seed, seq_seed, r_seed = seed_seq.generate_state(4)

                # sample pi for Q
                pi = rate_matrix_cpp.get_pi(pi_seed)

                # sample random reversible Q with init dist pi
                Q = rate_matrix_cpp.get_Q_from_pi(pi, q_seed)

                # sample ancestral sequence
                seq0 = rand_seq_cpp.rand_seq(n,freq,int(seq_seed))

                # sample site heterogeneity parameters
                rng = np.random.default_rng(r_seed)
                rs = rng.gamma(shape=alpha, scale=1/alpha, size=len(orf_lens))

                seeds = seed_seq.generate_state(len(orf_lens)+1)
                seq1 = ""
                for s, orf_len in enumerate(orf_lens):
                    P = mat_exp_cpp.mat_exp_4d(rs[s]*Q, t)
                    seq1 += seq_sim_cpp.seq_sim(seq0[orf_start[s]:orf_end[s]],P,int(seeds[s]))

                # last bit 
                P = mat_exp_cpp.mat_exp_4d(Q, t)
                seq1 += seq_sim_cpp.seq_sim(seq0[orf_end[s]:],P,int(seeds[s+1]))

                # calculate sample diversity matrix
                H = div_mat_cpp.div_mat(seq0, seq1)
                
                # calculate statistics 
                stat_b, df_b = bowker_stat2(H)
                stat_s = stuart_from_counts(H)
                stat_qs = qs_test(H, n, 3)
                P_hat = H / n
                d = get_d(H)

                d1 = safe_standardize(d[0], get_var(P_hat, n, 1))
                d2 = safe_standardize(d[1], get_var(P_hat, n, 2))
                d3 = safe_standardize(d[2], get_var(P_hat, n, 3))
                d4 = safe_standardize(d[3], get_var(P_hat, n, 4))

                i += 1
                #res_max_db_deviation.append(return_detailed_balance(Q,pi))
                results.append({
                    "alpha_gamma": alpha, 
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
                pbar.update(1)
            

    pbar.close()

    script_dir = Path(__file__).resolve().parent         
    out_dir = ROOT2 / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "revision/result_mixture_r.csv"

    # Save as one-column CSV
    #pd.DataFrame({"t": res_t, "Bowker": res_b, "Stuart": res_s, "QS": res_qs, "d1": res_d1, "d2": res_d2, "d3": res_d3, "d4": res_d4, "mut_sum": res_mut, "alpha_gamma": res_alpha}).to_csv(out_path, index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)

    print(f"Saved {len(results_df)} sims to: {out_path}")


if __name__ == "__main__":
    main()