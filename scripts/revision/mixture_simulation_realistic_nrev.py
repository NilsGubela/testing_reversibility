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
    # stationary distribution = whole genome nuc dist
    #counts = Counter(seq0)
    #freq = np.array([counts[nuc]/len(seq0) for nuc in nuc_order])
    # stationary distribution = spike region nuc dist
    spike = seq0[21563-1:25384]
    counts = Counter(spike)
    freq = np.array([counts[nuc]/len(spike) for nuc in nuc_order])

    orf_start = [266,13469,21563,25393,26245,26523,27202,27394,27894,28274,29558, 1]
    orf_end = [13468,21555,25384,26220,26472,27191,27387,27759,28259,29533,29674, len(reference_record.seq)+1]
    orf_label = ["1a", "1b", "S", "3a", "E", "M", "6", "7a", "8", "N", "10", "full", "rest"]

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


    k = 1000

    print("Length of seq0: ",len(seq0))
    print("Nucleotide dist: ", freq)

    seq0 = seq0.upper()
    nuc_order = [x.upper() for x in nuc_order]

    n = len(seq0)
    orf_lens = [13202, 8086, 3822, 828, 228, 669, 186, 336, 336, 1260, 117]
    #orf_a = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    orf_stat = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    orf_rev = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    orf_start = np.cumsum(orf_lens) - orf_lens
    orf_end = np.cumsum(orf_lens)
    
    i = 0
    results = []

    master_seed = 12345

    #ts = [0.01, 0.0215, 0.05, 0.1, 0.215, 0.5]
    #ts = [0.00215, 0.005]
    #ts = [0.0001, 0.00215]
    ts = [1]
    #alphas = [0.1, 0.5, 5]
    pbar = tqdm(total=k*len(ts), desc="Simulations", unit="rep")
    for t in ts:
        count = 0
        while count < k:
            # takeing care of seeds
            seed_seq = np.random.SeedSequence(
            [master_seed, i]
            )

            pi_seed, q_seed, r_seed = seed_seq.generate_state(3)

            # sample one nrev Q
            Qnrev = rate_matrix_cpp.get_Q_nonrev(q_seed)

            # sample pi for Q
            #pi = rate_matrix_cpp.get_pi(pi_seed)

            # sample random reversible Q with init dist like spike
            #Q = rate_matrix_cpp.get_Q_from_pi(freq, q_seed)

            # sample site heterogeneity parameters
            #rng = np.random.default_rng(r_seed)
            #rs = rng.gamma(shape=alpha, scale=1/alpha, size=len(orf_lens))
            rs = np.ones(len(orf_lens))
            #rs[2] = 4
            #rs[0] = 1/2
            #rs[1] = 1/2

            t = np.random.rand()*0.005


            seeds = seed_seq.generate_state(len(orf_lens)+1)
            seq1 = ""
            for s, orf_len in enumerate(orf_lens):
                seq0_tmp = seq0[orf_start[s]:orf_end[s]]

                seed_seq = np.random.SeedSequence(
                [master_seed, i]
                )

                pi_seed, q_seed, mut_seed = seed_seq.generate_state(3)

                if orf_stat[s] == 1:
                    counts = Counter(seq0_tmp)
                    freq = np.array([counts[nuc]/len(seq0_tmp) for nuc in nuc_order])
                else:
                    freq = np.array([[0.1],[0.1],[0.1],[0.7]])
                    #freq = rate_matrix_cpp.get_pi(pi_seed)

                if orf_rev[s] == 1:
                    a = np.random.rand()
                    Q = rate_matrix_cpp.get_HKY_from_pi_a(freq, a, q_seed)
                else:
                    Q = Qnrev

                #counts = Counter(seq0_tmp)
                #freq = np.array([counts[nuc]/len(seq0_tmp) for nuc in nuc_order])


                #Q = np.array([[-1, b, a, b],
                #                [b, -1, b, a],
                #                [a, b, -1, b],
                #                [b, a, b, -1]])

                #freq = np.array([[0.1],[0.4],[0.1],[0.4]])
                #Q = rate_matrix_cpp.get_HKY_from_pi_a(freq,a, q_seed)

                P = mat_exp_cpp.mat_exp_4d(rs[s]*Q, t)
                #print(orf_label[s], np.linalg.norm(np.matmul(freq.T, P) - freq.T))
                seq1_tmp = seq_sim_cpp.seq_sim(seq0_tmp,P,int(mut_seed))
                H = div_mat_cpp.div_mat(seq0_tmp, seq1_tmp)
                H_dict = {f"H_{nuc_order[i]}_{nuc_order[j]}": float(H[i, j]) for i in range(H.shape[0]) for j in range(H.shape[1])}
                n = len(seq0_tmp)
                # calculate statistics 
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
                    "seq_id": count,
                    "region": orf_label[s], 
                    "t": t,
                    "mut_sum": H.sum() - np.trace(H),
                    "stat_b": stat_b,
                    "stat_s": stat_s,
                    #"stat_qs": stat_qs,
                    "bowker_df": df_b,
                    "d1": d1,
                    "d2": d2,
                    "d3": d3,
                    "d4": d4,
                    **H_dict
                })
                seq1 += seq1_tmp
                i += 1

            # last bit 
            P = mat_exp_cpp.mat_exp_4d(Q, t)
            seq1 += seq_sim_cpp.seq_sim(seq0[orf_end[s]:],P,int(seeds[s+1]))

            # calculate sample diversity matrix
            H = div_mat_cpp.div_mat(seq0, seq1)
            H_dict = {f"H_{nuc_order[i]}_{nuc_order[j]}": float(H[i, j]) for i in range(H.shape[0]) for j in range(H.shape[1])}
            n = len(seq1)
            # calculate statistics 
            stat_b, df_b = bowker_stat2(H)
            stat_s = stuart_from_counts(H)
            #stat_qs = qs_test(H, n, 3)
            P_hat = H / n
            d = get_d(H)

            d1 = safe_standardize(d[0], get_var(P_hat, n, 1))
            d2 = safe_standardize(d[1], get_var(P_hat, n, 2))
            d3 = safe_standardize(d[2], get_var(P_hat, n, 3))
            d4 = safe_standardize(d[3], get_var(P_hat, n, 4))

            #res_max_db_deviation.append(return_detailed_balance(Q,pi))
            results.append({
                "seq_id": count,
                "region": "full", 
                "t": t,
                "mut_sum": H.sum() - np.trace(H),
                "stat_b": stat_b,
                "stat_s": stat_s,
                #"stat_qs": stat_qs,
                "bowker_df": df_b,
                "d1": d1,
                "d2": d2,
                "d3": d3,
                "d4": d4,
                **H_dict, 
            })

            i += 1
            count += 1
            pbar.update(1)

            

    pbar.close()

    script_dir = Path(__file__).resolve().parent         
    out_dir = ROOT2 / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "revision/result_mixture_realistic_nrev.csv"

    # Save as one-column CSV
    #pd.DataFrame({"t": res_t, "Bowker": res_b, "Stuart": res_s, "QS": res_qs, "d1": res_d1, "d2": res_d2, "d3": res_d3, "d4": res_d4, "mut_sum": res_mut, "alpha_gamma": res_alpha}).to_csv(out_path, index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)

    print(f"Saved {len(results_df)} sims to: {out_path}")


if __name__ == "__main__":
    main()