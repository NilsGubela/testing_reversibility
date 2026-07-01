#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
using namespace Eigen;
using namespace std;

Matrix4d mat_exp_4d(Matrix4d A, double t)
{
 EigenSolver<Matrix4d> es(A);
 
 Matrix4d D = es.pseudoEigenvalueMatrix();
 Matrix4d V = es.pseudoEigenvectors();
// take exp of D
 for(int i = 0; i < 4; i++){
    D(i,i) = exp(D(i,i) * t);
 }

 return V*D*V.inverse();
}

Eigen::Matrix<double, 4, 1> stat_dist(Eigen::Matrix<double, 4, 4> A)
{
   Eigen::Matrix<double, 4, 1> out;
   EigenSolver<Matrix4d> es(A.transpose());
   Eigen::Matrix<double, 4, 4>D = es.pseudoEigenvalueMatrix();
   Eigen::Matrix<double, 4, 4> V = es.pseudoEigenvectors();
   int i = 0;


   while( abs(D(i,i)) >  0.001){
      i +=1 ;
   }

   out = V(all,i);

  
   // normalize
   float sum = 0;

   for (int j = 0; j < 4; j++){
      sum += out(j,0);
   }


   out = 1/sum * out;

   return(out);

}


Eigen::Matrix<double, 4, 1> stat_distP(Eigen::Matrix<double, 4, 4> A)
{
   Eigen::Matrix<double, 4, 1> out;
   EigenSolver<Matrix4d> es(A.transpose());
   Eigen::Matrix<double, 4, 4>D = es.pseudoEigenvalueMatrix();
   Eigen::Matrix<double, 4, 4> V = es.pseudoEigenvectors();
   int i = 0;


   while( abs(D(i,i)-1) >  0.001){
      i +=1 ;
   }

   out = V(all,i);

  
   // normalize
   float sum = 0;

   for (int j = 0; j < 4; j++){
      sum += out(j,0);
   }


   out = 1/sum * out;

   return(out);

}


