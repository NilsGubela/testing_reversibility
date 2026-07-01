#include <cstdint>
#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>

namespace py = pybind11;

Eigen::Matrix4d get_diag(const Eigen::Vector4d& d)
{
    return d.asDiagonal();
}

Eigen::Matrix4d get_Q(uint32_t seed = 42)
{
    std::mt19937 generator(seed);

    std::uniform_real_distribution<double> dis(0.0, 1.0);

    Eigen::Vector4d d;
    Eigen::Vector4d row_sum;

    Eigen::Matrix4d psi;
    Eigen::Matrix4d S = Eigen::Matrix4d::Zero();
    Eigen::Matrix4d out;

    double sum = 0.0;

    // Sample stationary distribution
    for (int i = 0; i < 4; i++) {
        d(i) = dis(generator);
        sum += d(i);
    }

    d /= sum;

    psi = get_diag(d);

    // Symmetric exchangeability matrix
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {

            S(i,j) = dis(generator);

            S(j,i) = S(i,j);
        }
    }

    out = S * psi;

    row_sum = out.rowwise().sum();

    sum = 0.0;

    // Set diagonal entries
    for (int i = 0; i < 4; i++) {

        out(i,i) = -row_sum(i);

        sum -= out(i,i) * d(i);
    }

    // Normalize mean substitution rate to 1
    return out / sum;
}

Eigen::Matrix4d get_Q_nonrev(uint32_t seed = 42)
{
    std::mt19937 generator(seed);

    std::uniform_real_distribution<double> dis(0.0, 1.0);

    Eigen::Vector4d d;
    Eigen::Vector4d row_sum;

    Eigen::Matrix4d psi;
    Eigen::Matrix4d S = Eigen::Matrix4d::Zero();
    Eigen::Matrix4d out;

    double sum = 0.0;

    // Sample stationary distribution
    for (int i = 0; i < 4; i++) {

        d(i) = dis(generator);

        sum += d(i);
    }

    d /= sum;

    psi = get_diag(d);

    // Nonreversible constrained matrix
    S(0,1) = dis(generator);
    S(0,2) = dis(generator);
    S(0,3) = dis(generator);

    S(1,0) = dis(generator);
    S(1,2) = dis(generator);
    S(1,3) = dis(generator);

    S(2,0) = S(0,2);
    S(2,1) = S(0,3);
    S(2,3) = S(0,1);

    S(3,0) = S(1,2);
    S(3,1) = S(1,3);
    S(3,2) = S(1,0);

    out = S * psi;

    row_sum = out.rowwise().sum();

    sum = 0.0;

    // Set diagonal entries
    for (int i = 0; i < 4; i++) {

        out(i,i) = -row_sum(i);

        sum -= out(i,i) * d(i);
    }

    // Normalize mean substitution rate
    return out / sum;
}

PYBIND11_MODULE(rate_matrix_cpp, m)
{
    m.doc() = "Random reversible and nonreversible rate matrices";

    m.def(
        "get_Q",
        &get_Q,
        py::arg("seed") = 42,
        "Generate reversible 4x4 rate matrix"
    );

    m.def(
        "get_Q_nonrev",
        &get_Q_nonrev,
        py::arg("seed") = 42,
        "Generate nonreversible 4x4 rate matrix"
    );
}