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
    t1 = 0.5

    freq = np.array([[0.15],[0.1],[0.4],[0.35]])
    Q = np.array([ # HKY
    [-1.0,  0.2,  0.6,  0.2],
    [ 0.3, -1.0,  0.3,  0.4],
    [ 0.6,  0.2, -1.0,  0.2],
    [ 0.3,  0.4,  0.3, -1.0],
    ]) 
    # calculate P1
    P1 = mat_exp_cpp.mat_exp_4d(Q, t1)

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

        seq_seed, mut1_seed, mut2_seed, t_seed = seed_seq.generate_state(4)

        # sample ancestral sequence
        seq0 = rand_seq_cpp.rand_seq(n,freq,int(seq_seed))

        # sample time 
        rng = np.random.default_rng(t_seed)
        t2 = rng.uniform(0, 1)

        # calculate P
        P2 = mat_exp_cpp.mat_exp_4d(Q, t2)

        # simulate evolution 1
        seq1 = seq_sim_cpp.seq_sim(seq0,P1,int(mut1_seed))

        # simulate evolution 2
        seq2 = seq_sim_cpp.seq_sim(seq0,P2,int(mut2_seed))

        # calculate sample diversity matrix of cherry
        H = div_mat_cpp.div_mat(seq1, seq2)

        # check if matrix has zero non-diagonal entry
        mat_no_diag = H.copy()
        np.fill_diagonal(mat_no_diag, np.nan)
        if np.any(mat_no_diag == 0):
            stat_b = np.nan
            stat_s = np.nan
            stat_qs = np.nan
            #for s in range(4):
            #    res_sm[s].append(np.nan)
        
        else:

            m = get_m(H)
            B = get_B(H)
            stat_b = bowker_stat(m,B)
            stat_s = stuart_from_counts(H)
            stat_qs = qs_test(H, n)

        i += 1
        res_t.append(t2)
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

    out_path = out_dir / "figure1/result_nonstationary_reversible_cherry.csv"

    pd.DataFrame({"t2": res_t, "Bowker": res_b, "Stuart": res_s, "QS": res_qs, "mut_sum": res_mut}).to_csv(out_path, index=False)
    print(f"Saved {len(res_b)} sims to: {out_path}")


if __name__ == "__main__":
    main()