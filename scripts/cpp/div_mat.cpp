#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>

namespace py = pybind11;

int nuc_to_index(char nuc)
{
    switch(nuc) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;

        default:
            throw std::runtime_error("Invalid nucleotide");
    }
}

Eigen::Matrix4d div_mat(
    const std::string& seq1,
    const std::string& seq2
)
{
    if (seq1.size() != seq2.size()) {
        throw std::runtime_error(
            "Sequences must have equal length"
        );
    }

    Eigen::Matrix4d out = Eigen::Matrix4d::Zero();

    size_t n = seq1.size();

    for (size_t i = 0; i < n; i++) {

        int r = nuc_to_index(seq1[i]);

        int c = nuc_to_index(seq2[i]);

        out(r, c) += 1.0;
    }

    return out;
}

PYBIND11_MODULE(div_mat_cpp, m)
{
    m.doc() = "Sequence divergence matrix";

    m.def(
        "div_mat",
        &div_mat,
        py::arg("seq1"),
        py::arg("seq2"),
        "Compute 4x4 nucleotide divergence matrix"
    );
}