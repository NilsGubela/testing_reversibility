#include <string>
#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <cstdint>
#include <Eigen/Dense>

namespace py = pybind11;

int nuc_to_index(char nuc)
{
    switch(nuc) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default: return 0;
    }
}

char sample_nuc(double r, const Eigen::Vector4d& probs)
{
    if (r < probs(0)) {
        return 'A';

    } else if (r < probs(0) + probs(1)) {
        return 'C';

    } else if (r < probs(0) + probs(1) + probs(2)) {
        return 'G';

    } else {
        return 'T';
    }
}

std::string seq_sim(
    const std::string& seq,
    const Eigen::Matrix4d& P,
    uint32_t seed = 1
)
{
    std::mt19937 generator(seed);

    std::uniform_real_distribution<double> dis(0.0, 1.0);

    int n = seq.length();

    std::string seq_out;
    seq_out.reserve(n);

    for (int i = 0; i < n; i++) {

        int idx = nuc_to_index(seq[i]);

        double r = dis(generator);

        Eigen::Vector4d probs = P.row(idx);

        seq_out += sample_nuc(r, probs);
    }

    return seq_out;
}

PYBIND11_MODULE(seq_sim_cpp, m)
{
    m.def(
        "seq_sim",
        &seq_sim,
        py::arg("seq"),
        py::arg("P"),
        py::arg("seed") = 1
    );
}