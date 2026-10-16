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

    # non-reversible
    #Q = 1/(8.7692316)*np.array([[-6, 1, 2, 3], [4, -15, 5, 6],[2, 3, -6, 1],[5, 6, 4, -15]])

    # stationary
    #freq = np.array([[0.3461538],[0.1538462],[0.3461538],[0.1538462]])

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

        q_seed, seq_seed, mut_seed, t_seed = seed_seq.generate_state(4)

        # sample random UNREST
        Q = rate_matrix_cpp.get_Q_nonrev(q_seed)

        # get stationary distribution
        vals, vecs = np.linalg.eig(Q.T)
        pi = np.real(vecs[:, np.argmin(np.abs(vals))])
        pi = pi / pi.sum()   # normalize
        if return_detailed_balance(Q, pi) < 1:
            # sampel again
            i += 1
            continue

        # sample ancestral sequence
        seq0 = rand_seq_cpp.rand_seq(n,pi,int(seq_seed))

        # sample time 
        rng = np.random.default_rng(t_seed)
        #t = rng.uniform(0, 1) #* 0.01
        #a = 1e-6
        # logarithmically uniform spaced in (a,1)
        #t = 10 ** (np.log10(a) + ((count + 0.5) / k) * (-np.log10(a)))
        t = count/k
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

        results.append({
            "t": t,
            "mut_sum": H.sum() - np.trace(H),
            "stat_b": stat_b,
            "stat_s": stat_s,
            "bowker_df": df_b,
            "d1": d1,
            "d2": d2,
            "d3": d3,
            "d4": d4
        })

        i += 1
        count += 1
        pbar.update(1)
        pbar.set_postfix(rejects=i - count)

    pbar.close()

    script_dir = Path(__file__).resolve().parent            
    out_dir = ROOT2 / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "figure1/result_stationary_nonreversible.csv"

    #pd.DataFrame({"t": res_t, "Bowker": res_b, "Stuart": res_s, "QS": res_qs, "QS3": res_qs3, "mut_sum": res_mut}).to_csv(out_path, index=False)
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)
    print(f"Saved {len(results_df)} sims to: {out_path}")


if __name__ == "__main__":
    main()