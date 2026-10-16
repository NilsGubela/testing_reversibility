from Bio import SeqIO
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
ROOT2 = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT2))
from util import *

sys.path.append(str(ROOT2 / "build"))

import div_mat2_cpp


ALIGNMENT_FILE = ROOT / "data/processed/aligned.fasta"
METADATA_FILE = ROOT / "data/raw/sequences.csv"

OUTPUT_FILE = ROOT / "results/figure2/all_codon_pos.csv"

LINEAGE_COLUMN = "Pangolin"
ACCESSION_COLUMN = "Accession"


meta = pd.read_csv(METADATA_FILE)

# filter for classifiable 
meta = meta[meta["Pangolin"] != "unclassifiable"]


# maps accession -> aligned sequence record
alignment_dict = {}
nuc_dict = {0: "A", 1: "C", 2: "G", 3: "T"}

for record in SeqIO.parse(ALIGNMENT_FILE, "fasta"):

    accession = record.id.split(".")[0]

    alignment_dict[accession] = record


# assumes first sequence is reference
reference_record = next(
    SeqIO.parse(ALIGNMENT_FILE, "fasta")
)

orf_start = [266,13469,21563,25393,26245,26523,27202,27394,27894,28274,29558]
orf_end = [13467,21554,25384,26220,26472,27191,27387,27759,28259,29533,29674]
orf_label = ["1ab", "S", "3a", "E", "M", "6", "7a", "8", "N", "10"]

codon_pos = [0, 1, 2, 3]

results = []


seq0_base = ""
for start, end in zip(orf_start, orf_end):
    seq0_base += str(reference_record.seq)[start-1:end]


pbar = tqdm(total=len(meta)*(len(codon_pos)), desc="Sequences", unit="seq")
for _, row in meta.iterrows():

    accession = row[ACCESSION_COLUMN]
    lineage = row[LINEAGE_COLUMN]

    if lineage == "unclassifiable":
        continue

    if accession not in alignment_dict:

        print(f"Missing alignment for {accession}")

        continue

    record = alignment_dict[accession]

    seq1_base = ""
    for start, end in zip(orf_start, orf_end):
        seq1_base += str(record.seq)[start-1:end]

    for codon in codon_pos:

        if codon == 3:
            seq0 = seq0_base
            seq1 = seq1_base
        else:

            seq0 = seq0_base[codon::3]
            seq1 = seq1_base[codon::3]

        # generate diversity matrix with reference 
        H = div_mat2_cpp.div_mat2(seq0, seq1)
        n = np.sum(H)
        stat_s = stuart_from_counts(H)
        #stat_qs = qs_test(H, n, 3)
        stat_b, df_b = bowker_stat2(H)

        P_hat = H / n
        d = get_d(H)

        if n <= 5:
            d1, d2, d3, d4 = np.nan, np.nan, np.nan, np.nan
        else:
            d1 = safe_standardize(d[0], get_var(P_hat, n, 1))
            d2 = safe_standardize(d[1], get_var(P_hat, n, 2))
            d3 = safe_standardize(d[2], get_var(P_hat, n, 3))
            d4 = safe_standardize(d[3], get_var(P_hat, n, 4))

        H_dict = {f"H_{nuc_dict[i]}_{nuc_dict[j]}": float(H[i, j]) for i in range(H.shape[0]) for j in range(H.shape[1])}

        results.append({
            "n": len(seq0),
            "n_used": n,
            "accession": accession,
            "pangolin_lineage": lineage,
            "mut_sum": H.sum() - np.trace(H),
            "stat_b": stat_b,
            "stat_s": stat_s,
            #"stat_qs": stat_qs,
            "bowker_df": df_b,
            "codon": codon,
            "d1": d1,
            "d2": d2,
            "d3": d3,
            "d4": d4,
            **H_dict,  
        })

        pbar.update(1)

pbar.close()


results_df = pd.DataFrame(results)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Saved results to:")
print(OUTPUT_FILE)