#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
mkdir -p "$BUILD_DIR"

PYTHON="$SCRIPT_DIR/../../.venv/bin/python"
CXX=c++

echo "Using Python: $PYTHON"
$PYTHON --version

CXXFLAGS="-O3 -Wall -shared -std=c++17 -fPIC"
PYBIND=$($PYTHON -m pybind11 --includes)
EXT=$($PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
PYTHON_INC=$($PYTHON -c 'import sysconfig; print("-I" + sysconfig.get_paths()["include"])')
EIGEN="-I/opt/homebrew/include/eigen3"

echo "PYBIND=$PYBIND"
echo "PYTHON_INC=$PYTHON_INC"
echo "EXT=$EXT"

compile_module () {
    SRC=$1
    NAME=$2

    echo "Compiling $SRC -> $BUILD_DIR/${NAME}${EXT}"

    $CXX $CXXFLAGS \
        -undefined dynamic_lookup \
        $PYBIND \
        $PYTHON_INC \
        $EIGEN \
        "$SCRIPT_DIR/$SRC" \
        -o "$BUILD_DIR/${NAME}${EXT}"
}

compile_module rand_seq.cpp rand_seq_cpp
compile_module seq_sim.cpp seq_sim_cpp
compile_module mat_exp.cpp mat_exp_cpp
compile_module div_mat.cpp div_mat_cpp
compile_module div_mat2.cpp div_mat2_cpp
compile_module rate_matrix.cpp rate_matrix_cpp

echo "Done."
ls -lah "$BUILD_DIR"