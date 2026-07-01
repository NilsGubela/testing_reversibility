#include<iostream>
#include <string>
#include <map>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <random>
using namespace std;


string seq_sim(string seq, map<string , map <string , float> > P, std::mt19937& generator, std::uniform_real_distribution<double> dis)
{
	int n = seq.length();
	map<string, float> nuc_freq;
	string nuc;
	double r;
	string seq_out = "";
	string nuc_out;


	for(int i = 0; i < n; i++){
		nuc = seq[i];
		nuc_freq = P[nuc];
		r = dis(generator);
		if (r < nuc_freq["A"]){
  			nuc_out = 'A';
  		} else if (nuc_freq["A"] <= r && r <nuc_freq["A"]+nuc_freq["C"]){
  			nuc_out = 'C';
  		} else if (nuc_freq["A"]+nuc_freq["C"] <= r && r <nuc_freq["A"]+nuc_freq["C"]+nuc_freq["G"]){
  			nuc_out = 'G';
  		}else {
  			nuc_out = 'T';
  		}
  		seq_out += nuc_out;
	}
	return(seq_out);
}

string seq_sim_new(string seq,  Eigen::Matrix4d P)
{
	int n = seq.length();
	map<string, int> nuc_dict;
	nuc_dict["A"] = 0;
	nuc_dict["C"] = 1;
	nuc_dict["G"] = 2;
	nuc_dict["T"] = 3;

	string nuc;
	double r;
	int nuc_index;
	string seq_out = "";
	string nuc_out;


	for(int i = 0; i < n; i++){
		nuc = seq[i];
		nuc_index = nuc_dict[nuc];
		r = ((double) rand() / (RAND_MAX));
		if (r < P(nuc_index,0)){
  			nuc_out = "A";
  		} else if (r < P(nuc_index,0)+P(nuc_index,1)){
  			nuc_out = "C";
  		} else if (r < P(nuc_index,0)+P(nuc_index,1)+P(nuc_index,2)){
  			nuc_out = "G";
  		}else {
  			nuc_out = "T";
  		}
  		seq_out += nuc_out;
	}
	return(seq_out);
}