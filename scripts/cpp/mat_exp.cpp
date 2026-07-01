#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace py = pybind11;

Eigen::Matrix4d mat_exp_4d(
    const Eigen::Matrix4d& A,
    double t
)
{
    return (A * t).exp();
}

PYBIND11_MODULE(mat_exp_cpp, m)
{
    m.doc() = "Matrix exponential for 4x4 matrices using Eigen";

    m.def(
        "mat_exp_4d",
        &mat_exp_4d,
        py::arg("A"),
        py::arg("t"),
        "Compute exp(A * t)"
    );
}