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

    nuc_order = ["A", "C", "G", "T"]

    t = 0.5

    count = 0
    i = 0
    results = []

    master_seed = 12345

    pbar = tqdm(total=k, desc="Simulations", unit="rep")

    while count < k:
        # takeing care of seeds
        seed_seq = np.random.SeedSequence(
        [master_seed, i]
        )

        pi_seed, q_seed, seq_seed, mut_seed = seed_seq.generate_state(4)

        # sample pi for Q
        pi = rate_matrix_cpp.get_pi(pi_seed)

        # sample random reversible Q with init dist pi
        #Q = rate_matrix_cpp.get_Q_from_pi(pi, q_seed)
        a = np.random.rand()
        Q = rate_matrix_cpp.get_HKY_from_pi_a(pi,a, q_seed)

        # sample ancestral sequence
        seq0 = rand_seq_cpp.rand_seq(n,freq,int(seq_seed))

        # calculate P
        P = mat_exp_cpp.mat_exp_4d(Q, t)

        # simulate evolution
        seq1 = seq_sim_cpp.seq_sim(seq0,P,int(mut_seed))

        # calculate sample diversity matrix
        H = div_mat_cpp.div_mat(seq0, seq1)

        stat_b, df_b = bowker_stat2(H)
        stat_s = stuart_from_counts(H)
        #stat_qs = qs_test(H, n, 3)
        P_hat = H / n
        d = get_d(H)
        d1 = safe_standardize(d[0], get_var(P_hat, n, 1))
        d2 = safe_standardize(d[1], get_var(P_hat, n, 2))
        d3 = safe_standardize(d[2], get_var(P_hat, n, 3))
        d4 = safe_standardize(d[3], get_var(P_hat, n, 4))
        H_dict = {f"H_{nuc_order[i]}_{nuc_order[j]}": float(H[i, j]) for i in range(H.shape[0]) for j in range(H.shape[1])}
        results.append({
            "t": t,
            "mut_sum": H.sum() - np.trace(H),
            "stat_b": stat_b,
            "stat_s": stat_s,
            "bowker_df": df_b,
            "d1": d1,
            "d2": d2,
            "d3": d3,
            "d4": d4,
            **H_dict
        })

        i += 1
        count += 1
        pbar.update(1)
        pbar.set_postfix(rejects=count)

    pbar.close()

    script_dir = Path(__file__).resolve().parent         
    out_dir = ROOT2 / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "revision/result_ababneh_nonstationary_reversible.csv"

    # Save as one-column CSV
    #pd.DataFrame({"t": res_t, "Bowker": res_b, "Stuart": res_s, "QS": res_qs, "QS3": res_qs3, "d1": res_d1, "d2": res_d2, "d3": res_d3, "d4": res_d4, "mut_sum": res_mut}).to_csv(out_path, index=False)
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)


    print(f"Saved {len(results)} sims to: {out_path}")


if __name__ == "__main__":
    main()