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
compile_module div_mat2.cpp div_mat2_cpp
compile_module rate_matrix.cpp rate_matrix_cpp
