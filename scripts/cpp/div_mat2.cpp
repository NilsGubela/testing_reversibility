#include <stdexcept>
#include <string>
#include <cctype>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>

namespace py = pybind11;


// ============================================================
// nucleotide -> index
// returns:
//   0=A
//   1=C
//   2=G
//   3=T
//  -1=invalid
// ============================================================

int nuc_to_index(char nuc)
{
    nuc = std::tolower(nuc);

    switch(nuc) {

        case 'a': return 0;
        case 'c': return 1;
        case 'g': return 2;
        case 't': return 3;

        default:
            return -1;
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

        // skip invalid nucleotides
        if (r == -1 || c == -1) {
            continue;
        }

        out(r, c) += 1.0;
    }

    return out;
}


PYBIND11_MODULE(div_mat2_cpp, m)
{
    m.doc() = "Sequence divergence matrix";

    m.def(
        "div_mat2",
        &div_mat,
        py::arg("seq1"),
        py::arg("seq2"),
        "Compute 4x4 nucleotide divergence matrix"
    );
}