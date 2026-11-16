![](https://img.shields.io/github/v/release/NilsGubela/testing_reversibility)

# Testing reversibility in viral evolution

This repository contains codes to reproduce the manuscript "A statistical framework to distinguish stationarity from reversibility in viral evolution". 
The code for the simulation studies is found at [scripts/simulation](https://github.com/NilsGubela/testing_reversibility/tree/main/scripts/simulation) and scripts for the analysis of SARS-CoV-2 genomic sequences at [scripts/sequence_analysis](https://github.com/NilsGubela/testing_reversibility/tree/main/scripts/sequence_analysis). Functions are written in C++ and compiled to python modules.

---

# Setting Up the Virtual Environment

The code was developed for Python 3 and macOS. Install Eigen separately; the build script expects it at `/opt/homebrew/include/eigen3`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  numpy scipy matplotlib pandas biopython pybind11 tqdm tabulate
brew install eigen
```

# Compiling the C++ Extensions

The Python scripts import extensions from `scripts/build/`.

```bash
chmod +x scripts/cpp/build.sh
scripts/cpp/build.sh
PYTHONPATH=scripts/build python scripts/cpp/test.py
```
## Run the simulations

Each command writes its CSV to `results/simulation/`. These jobs are computationally intensive; the scripts use fixed master seeds (`12345`) for reproducibility.

```bash
python scripts/simulation/lin_nons_rev.py
python scripts/simulation/parallel_nons_rev.py
python scripts/simulation/lin_s_nonrev.py
python scripts/simulation/parallel_s_nonrev.py
python scripts/simulation/ababneh_nonstationary_reversible_f81.py
python scripts/simulation/mixture_simulation_stat_rev.py
python scripts/simulation/mixture_simulation_sitehet.py
python scripts/simulation/mixture_simulation_gtr.py
```

The first four scripts generate the stationary/nonstationary and cherry-picking simulation results; the last four generate the mixture-model results. The corresponding files are:

```text
result_nonstationary_reversible.csv
result_nonstationary_reversible_cherry.csv
result_stationary_nonreversible.csv
result_stationary_nonreversible_cherry.csv
result_ababneh_nonstationary_reversible_f81.csv
result_mixture_stat_rev.csv
result_mixture_realistic_sitehet.csv
result_mixture_gtr.csv
```

## Run error analyses (type-1 & type-2)

```bash
python scripts/error_analysis/type1_error.py
python scripts/error_analysis/type2_error.py --vary divergence
python scripts/error_analysis/type2_error.py --vary sequence_length
python scripts/error_analysis/type2_error.py --vary db_deviation
```

These commands create `results/error_analysis/result_type1.csv`, `result_type2_divergence.csv`, `result_type2_length.csv`, and `result_type2_db_deviation.csv`.

## Prepare the SARS-CoV-2 data

The sequence analysis requires these files:

```text
data/raw/sequences.csv
data/raw/wuhan_reference.fasta
data/processed/sequences_filtered.fasta
data/processed/aligned.fasta
```

Download complete human SARS-CoV-2 nucleotide genomes from [NCBI Virus](https://www.ncbi.nlm.nih.gov/labs/virus/vssi/) using:

- lineage: SARS-CoV-2 (`taxid:2697049`)
- sequence type: nucleotide
- completeness: complete
- host: *Homo sapiens*
- country: Germany

Export the sequence metadata as `data/raw/sequences.csv` (including `Accession`, `Pangolin`, and `Collection_Date`) and the filtered FASTA sequences as `data/processed/sequences_filtered.fasta`. Download Wuhan-Hu-1 (`NC_045512.2`) as `data/raw/wuhan_reference.fasta`.

Install MAFFT and align the downloaded sequences while preserving reference coordinates:

```bash
conda install -c bioconda mafft
mkdir -p data/processed
mafft --keeplength --addfragments \
  data/processed/sequences_filtered.fasta \
  data/raw/wuhan_reference.fasta \
  > data/processed/aligned.fasta
```
## Analyze SARS-CoV-2 sequences

```bash
python scripts/sequence_analysis/test_all_orfs.py
```

This creates `results/sequence_analysis/all_orfs.csv` for the `1a/b`, spike (`S`), remaining, and full-genome regions.

## Recreate the manuscript figures

Run after the required simulation and sequence-analysis CSVs exist:

```bash
python scripts/plotting/figure1.py
python scripts/plotting/figure2.py
python scripts/plotting/figure3.py
python scripts/plotting/figure4.py
python scripts/plotting/figure5.py
python scripts/plotting/figureD1.py
```

The resulting PDFs are written to `figs/` as `figure1.pdf`–`figure5.pdf` and `figureD1.pdf`.



