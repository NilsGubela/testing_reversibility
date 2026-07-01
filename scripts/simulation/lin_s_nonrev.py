import numpy as np
import sys
from pathlib import Path
from util import *
from tqdm import tqdm
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROOT2 = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT / "build"))

import rand_seq_cpp
import seq_sim_cpp
import mat_exp_cpp
import div_mat_cpp
import rate_matrix_cpp


def main():
    n = 10000
    k = 10000

    # non-reversible
    Q = 1/(8.7692316)*np.array([[-6, 1, 2, 3], [4, -15, 5, 6],[2, 3, -6, 1],[5, 6, 4, -15]])

    # stationary
    freq = np.array([[0.3461538],[0.1538462],[0.3461538],[0.1538462]])

    count = 0
    i = 0
    res_t = []
    res_b = []
    res_s = []
    res_qs = []
    res_mut = []


    master_seed = 12345

    pbar = tqdm(total=k, desc="Simulations", unit="rep")

    while count < k:
        # takeing care of seeds
        seed_seq = np.random.SeedSequence(
        [master_seed, i]
        )

        seq_seed, mut_seed, t_seed = seed_seq.generate_state(3)

        # sample ancestral sequence
        seq0 = rand_seq_cpp.rand_seq(n,freq,int(seq_seed))

        # sample time 
        rng = np.random.default_rng(t_seed)
        t = rng.uniform(0, 1) #* 0.01

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
            stat_b = np.nan
            stat_s = np.nan
            stat_qs = np.nan

        else:

            m = get_m(H)
            B = get_B(H)
            stat_b = bowker_stat(m,B)
            stat_s = stuart_from_counts(H)
            stat_qs = qs_test(H, n)

        i += 1
        res_t.append(t)
        res_b.append(stat_b)
        res_s.append(stat_s)
        res_qs.append(stat_qs)
        np.fill_diagonal(mat_no_diag, 0)
        res_mut.append(np.sum(mat_no_diag))
        count += 1
        pbar.update(1)
        pbar.set_postfix(rejects=i - count)

    pbar.close()

    script_dir = Path(__file__).resolve().parent            
    out_dir = ROOT2 / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "figure1/result_stationary_nonreversible.csv"

    pd.DataFrame({"t": res_t, "Bowker": res_b, "Stuart": res_s, "QS": res_qs, "mut_sum": res_mut}).to_csv(out_path, index=False)
    print(f"Saved {len(res_b)} sims to: {out_path}")


if __name__ == "__main__":
    main()