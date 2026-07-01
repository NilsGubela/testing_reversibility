from Bio import SeqIO
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from util import *
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
ROOT2 = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT2 / "build"))

import div_mat2_cpp


ALIGNMENT_FILE = ROOT / "data/processed/aligned.fasta"
METADATA_FILE = ROOT / "data/raw/sequences.csv"

OUTPUT_FILE = ROOT / "results/figure2/all_seq_all_regions.csv"

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

seq0 = str(reference_record.seq)



# go through every entry in meta, get the sequence and run all tests
results = []

pbar = tqdm(total=len(meta), desc="Sequences", unit="seq")
for _, row in meta.iterrows():

    accession = row[ACCESSION_COLUMN]
    lineage = row[LINEAGE_COLUMN]

    if lineage == "unclassifiable":
        continue

    if accession not in alignment_dict:

        print(f"Missing alignment for {accession}")

        continue

    record = alignment_dict[accession]
    seq1 = str(record.seq)
    n = len(seq1)

    # generate diversity matrix with reference 
    H = div_mat2_cpp.div_mat2(seq0, seq1)
    try:
        m = get_m(H)
        B = get_B(H)
        stat_b = bowker_stat(m,B)
    except:
        stat_b = np.nan
    stat_s = stuart_from_counts(H)
    stat_qs = qs_test(H, n)
    stat_c = saturation(H)

    H_dict = {f"H_{nuc_dict[i]}_{nuc_dict[j]}": float(H[i, j]) for i in range(H.shape[0]) for j in range(H.shape[1])}

    results.append({
        "accession": accession,
        "pangolin_lineage": lineage,
        "mut_sum": H.sum() - np.trace(H),
        "stat_b": stat_b,
        "stat_s": stat_s,
        "stat_qs": stat_qs,
        "stat_c": stat_c,
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