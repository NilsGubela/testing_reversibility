# Quasi-Symmetry Simulation Framework

This repository contains a hybrid Python/C++ framework for simulating nucleotide substitution processes and evaluating quasi-symmetry statistics.

The project combines:

- **Python**
  - experiment orchestration
  - statistical analysis
  - plotting
  - notebooks

with

- **C++**
  - fast sequence simulation
  - matrix exponential computation
  - Markov model simulation
  - pybind11 bindings

The C++ components are exposed to Python using `pybind11`.

---

# Repository Structure

```text
quasi_symmetry/
│
├── cpp/                # C++ source files + build script
├── build/              # Compiled Python extension modules (.so)
├── scripts/
│   └── python/
│       ├── experiments/
│       ├── analysis/
│       └── notebooks/
│
├── data/
│
├── requirements.txt
└── README.md
```

---

# Requirements

## macOS

Install:

- Homebrew
- Python 3.14+
- Eigen

### Install Homebrew

See:

https://brew.sh/

### Install Python

```bash
brew install python
```

### Install Eigen

```bash
brew install eigen
```

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

---

# Example build.sh

```bash
#!/bin/bash

mkdir -p ../build

CXXFLAGS="-O3 -Wall -shared -std=c++17"

PYBIND="$(python -m pybind11 --includes)"

PYTHON_INC="$(python3-config --includes)"

EXT="$(python3-config --extension-suffix)"

EIGEN="-I/opt/homebrew/include/eigen3"

compile_module () {
    SRC=$1
    NAME=$2

    c++ $CXXFLAGS \
        -undefined dynamic_lookup \
        $PYBIND \
        $PYTHON_INC \
        $EIGEN \
        $SRC \
        -o ../build/${NAME}${EXT}
}

compile_module rand_seq.cpp rand_seq_cpp
compile_module seq_sim.cpp seq_sim_cpp
compile_module mat_exp.cpp mat_exp_cpp
compile_module div_mat.cpp div_mat_cpp
compile_module rate_matrix.cpp rate_matrix_cpp
```

---

# Important Notes About Python Versions

The Python version used for:

- creating the virtual environment
- compiling the C++ extensions
- running the Python scripts

MUST be the same.

Verify:

```bash
python --version
```

and:

```bash
python3-config --extension-suffix
```

Example:

```text
Python 3.14
.cpython-314-darwin.so
```

---

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

# Importing the Compiled Modules

Python scripts should add the `build/` directory to the path.

Example:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

sys.path.append(str(ROOT / "build"))

import rand_seq_cpp
import seq_sim_cpp
import mat_exp_cpp
```

---

# Development Workflow

Typical workflow:

```bash
source .venv/bin/activate

cd cpp
./build.sh

cd ../scripts/python/experiments

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

## Notes

* `--addfragments` aligns query sequences to the reference without altering the reference alignment.
* `--keeplength` preserves the reference genome length and removes insertions relative to the reference.
* For reproducibility, it is recommended to record the MAFFT version used for the alignment.


# Notes

- Heavy numerical computations are implemented in C++ for speed.
- Python is used for high-level experiment management and analysis.
- Matrix operations use Eigen.
- Python bindings use pybind11.