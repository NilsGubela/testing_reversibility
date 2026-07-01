#include<iostream>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <random>
using namespace std;
 

string rand_seq(int n, Eigen::Matrix<double, 4, 1> freq, std::mt19937& generator, std::uniform_real_distribution<double> dis) {
	string seq = "";
	string nuc;
	for (int i = 0; i < n; i++) {
  		double r = dis(generator);
  		if (r < freq(0,0)){
  			nuc = 'A';
  		} else if (r <freq(0,0)+freq(1,0)){
  			nuc = 'C';
  		} else if (r <freq(0,0)+freq(1,0)+freq(2,0)){
  			nuc = 'G';
  		}else {
  			nuc = 'T';
  		}

  		seq += nuc;
}
	return seq;
}