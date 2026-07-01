# Testing reversibility in viral evolution

This repository contains codes to reproduce the manuscript "A statistical framework to distinguish stationarity from reversibility in viral evolution". 
The code for the simulation studies is found at [scripts/simulation](https://github.com/NilsGubela/testing_reversibility/tree/main/scripts/simulation) and scripts for the analysis of SARS-CoV-2 genomic sequences at [scripts/sequence_analysis](https://github.com/NilsGubela/testing_reversibility/tree/main/scripts/sequence_analysis). Functions are written in C++ and compiled to python modules.

---

# Setting Up the Virtual Environment

From the project root:

```bash
python3 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

Verify:

```bash
which python
```

Expected output:

```text
.../quasi_symmetry/.venv/bin/python
```

---

# Installing Python Dependencies

Install required packages:

```bash
pip install \
    numpy \
    scipy \
    matplotlib \
    pandas \
    pybind11 \
    tqdm
```

---

# Compiling the C++ Extensions

Move into the C++ directory:

```bash
cd cpp
```

Make the build script executable:

```bash
chmod +x build.sh
```

Compile all extension modules:

```bash
./build.sh
```

Compiled modules will appear in:

```text
build/
```

Example:

```text
rand_seq_cpp.cpython-314-darwin.so
seq_sim_cpp.cpython-314-darwin.so
mat_exp_cpp.cpython-314-darwin.so
...
```

# Running Experiments

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Then run experiments:

```bash
cd scripts/python/experiments

python stationary_nonreversible.py
```

---

# Sequence Analysis

## Downloading SARS-CoV-2 Genome Sequences

Complete SARS-CoV-2 genomes from human hosts in Germany were downloaded from the NCBI Virus database. The following filters were applied:

* Sequence type: Nucleotide
* Completeness: Complete genomes only
* Host: Human (*Homo sapiens*)
* Country: Germany

The filtered dataset can be downloaded from:

https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/virus?SeqType_s=Nucleotide&VirusLineage_ss=taxid:2697049&Completeness_s=complete&HostLineage_ss=Homo%20sapiens%20(human),%20taxid:9606&BaselineSurveillance_s=include&Country_s=Germany

## Reference Genome

All sequences were aligned against the Wuhan-Hu-1 reference genome (NC_045512.2), available from NCBI:

https://www.ncbi.nlm.nih.gov/nuccore/NC_045512.2?report=fasta

Save the reference sequence as:

```text
data/raw/wuhan_reference.fasta
```

## Sequence Alignment

Sequence alignment was performed using MAFFT.

### Installation

Install MAFFT via Conda:

```bash
conda install -c bioconda mafft
```

### Alignment

The downloaded sequences were aligned to the reference genome using the `--addfragments` option while preserving the reference coordinate system with `--keeplength`:

```bash
mafft --keeplength --addfragments \
    data/processed/sequences_filtered.fasta \
    data/processed/wuhan_reference.fasta \
    > data/processed/aligned.fasta
```

The resulting alignment is written to:

```text
data/processed/aligned.fasta
```
