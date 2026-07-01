#include <string>
#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <cstdint>
#include <Eigen/Dense>

namespace py = pybind11;

std::string rand_seq(
    int n,
    Eigen::Matrix<double, 4, 1> freq,
    uint32_t seed = 42
) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    std::string seq = "";

    for (int i = 0; i < n; i++) {

        double r = dis(generator);

        if (r < freq(0,0)) {
            seq += 'A';

        } else if (r < freq(0,0) + freq(1,0)) {
            seq += 'C';

        } else if (r < freq(0,0) + freq(1,0) + freq(2,0)) {
            seq += 'G';

        } else {
            seq += 'T';
        }
    }

    return seq;
}

PYBIND11_MODULE(rand_seq_cpp, m) {

    m.def(
        "rand_seq",
        &rand_seq,
        py::arg("n"),
        py::arg("freq"),
        py::arg("seed") = 42
    );
}