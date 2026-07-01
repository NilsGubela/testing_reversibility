#include<iostream>
#include <string>
#include <map>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
using namespace std;


map<string , map <string , float> > p_to_map(Eigen::Matrix4d P)
{
	map<string , map <string , float> > out;
	map <string , float> nuc_freq;
	string nuc_array[4] = {"A", "C", "G", "T"};
	string nuc;
	string nuc2;

	for (int i = 0; i < 4; i++){
		nuc = nuc_array[i];
		for (int j = 0; j < 4; j++){
			nuc2 = nuc_array[j];
			nuc_freq[nuc2] = P(i,j);
		}
		out[nuc] = nuc_freq;
	}
	return(out);
}