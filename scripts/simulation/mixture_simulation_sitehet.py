import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from Bio import SeqIO
from collections import Counter

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
    ALIGNMENT_FILE = ROOT2 / "data/processed/aligned.fasta"
    nuc_order = ["a", "c", "g", "t"]

    # load sequence 
    reference_record = next(
        SeqIO.parse(ALIGNMENT_FILE, "fasta")
    )

    seq0 = str(reference_record.seq)

    orf_start = [266,21563, 1]
    orf_end = [21555,25384, len(reference_record.seq)+1]
    orf_label = ["1a/b", "S",  "full", "rest"]

    # rearange seq0 such that all components are next to each other
    seq0 = ""
    for i, label in enumerate(orf_label):
        if label == "full":
            continue
        elif label == "rest":
            start = 0  # python index
            for l, orf_start_1based in enumerate(orf_start):
                if orf_label[l] == "full":
                    break
                # append sequence before this ORF
                seq0 += str(reference_record.seq)[start : orf_start_1based - 1]
                # next non-ORF starts right after this ORF
                start = orf_end[l]
            # append final trailing non-ORF region
            seq0 += str(reference_record.seq)[start:]
        else:
            start = orf_start[i]
            end = orf_end[i]
            seq0 += str(reference_record.seq)[start - 1 : end]


    k = 500

    print("Length of seq0: ",len(seq0))

    seq0 = seq0.upper()
    nuc_order = [x.upper() for x in nuc_order]

    n = len(seq0)
    orf_lens = [13202+8086, 3822, 828+ 228+ 669+ 186+ 336+ 336+ 1260+ 117+ 833]
    orf_label = ["1a/b", "S", "rest"]

    orf_start = np.cumsum(orf_lens) - orf_lens
    orf_end = np.cumsum(orf_lens)
    
    i = 0
    results = []

    master_seed = 12345

    ts = [1]
    pbar = tqdm(total=k*len(ts), desc="Simulations", unit="rep")
    for t in ts:
        count = 0
        while count < k:
            # takeing care of seeds
            seed_seq = np.random.SeedSequence(
            [master_seed, i]
            )

            pi_seed = seed_seq.generate_state(1)

            # sample stationary dist for all Q
            pi = rate_matrix_cpp.get_pi(pi_seed)

            t = 0.05#np.random.rand()*0.1

            seq1 = ""
            for s, orf_len in enumerate(orf_lens):
                seq0_tmp = seq0[orf_start[s]:orf_end[s]]

                seed_seq = np.random.SeedSequence(
                [master_seed, i]
                )

                q_seed, mut_seed = seed_seq.generate_state(2)

                # get random reversible Q with prescribed stat dist 
                Q = rate_matrix_cpp.get_Q_from_pi(pi, q_seed)
                P = mat_exp_cpp.mat_exp_4d(Q, t)

                seq1_tmp = seq_sim_cpp.seq_sim(seq0_tmp,P,int(mut_seed))
                H = div_mat_cpp.div_mat(seq0_tmp, seq1_tmp)
                H_dict = {f"H_{nuc_order[i]}_{nuc_order[j]}": float(H[i, j]) for i in range(H.shape[0]) for j in range(H.shape[1])}
                n = len(seq0_tmp)
                # calculate statistics 
                stat_b, df_b = bowker_stat2(H)
                stat_s = stuart_from_counts(H)
                stat_qs, df_qs = qs_test_with_df(H,n)
                results.append({
                    "seq_id": count,
                    "region": orf_label[s], 
                    "t": t,
                    "mut_sum": H.sum() - np.trace(H),
                    "stat_b": stat_b,
                    "stat_s": stat_s,
                    "bowker_df": df_b,
                    "stat_qs": stat_qs,
                    "qs_df": df_qs,
                    **H_dict
                })
                seq1 += seq1_tmp
                i += 1

            # calculate sample diversity matrix
            H = div_mat_cpp.div_mat(seq0, seq1)
            H_dict = {f"H_{nuc_order[i]}_{nuc_order[j]}": float(H[i, j]) for i in range(H.shape[0]) for j in range(H.shape[1])}
            n = len(seq1)
            # calculate statistics 
            stat_b, df_b = bowker_stat2(H)
            stat_s = stuart_from_counts(H)

            stat_qs, df_qs = qs_test_with_df(H,n)
            results.append({
                "seq_id": count,
                "region": "full", 
                "t": t,
                "mut_sum": H.sum() - np.trace(H),
                "stat_b": stat_b,
                "stat_s": stat_s,
                "bowker_df": df_b,
                "stat_qs": stat_qs,
                "qs_df": df_qs,

                **H_dict, 
            })

            i += 1
            count += 1
            pbar.update(1)

            

    pbar.close()

    script_dir = Path(__file__).resolve().parent         
    out_dir = ROOT2 / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "simulation/result_mixture_realistic_sitehet.csv"

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)

    print(f"Saved {len(results_df)} sims to: {out_path}")


if __name__ == "__main__":
    main()