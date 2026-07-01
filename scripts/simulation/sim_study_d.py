import numpy as np
import sys
from pathlib import Path
from util import *
import argparse
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
    res = []

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

        # select random index for which statistic is calculated
        index = np.random.choice(4)

        # calculate statistic for index
        P_hat = H / n
        d = get_d(H)

        stat = d[index] / np.sqrt(get_var(P_hat, n, index + 1))

        i += 1
        if np.isnan(stat):
            
            pbar.set_postfix(rejects=i - count - 1)
            continue

        res.append(stat)
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
    pd.DataFrame({"stat": res}).to_csv(out_path, index=False)

    print(f"Saved {len(res)} stats to: {out_path}")


if __name__ == "__main__":
    main()