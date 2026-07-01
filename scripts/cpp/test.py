import numpy as np
import rand_seq_cpp
import seq_sim_cpp
import mat_exp_cpp
import div_mat_cpp

from scipy.linalg import expm

freq = np.array([
    [0.25],
    [0.25],
    [0.25],
    [0.25]
])

seq0 = rand_seq_cpp.rand_seq(20, freq, seed = 1)

print(seq0)


A = np.array([
    [-1.0, 0.3, 0.4, 0.3],
    [0.2, -1.0, 0.5, 0.3],
    [0.3, 0.3, -1.0, 0.4],
    [0.4, 0.2, 0.2, -0.8]
])

t = 10

P = mat_exp_cpp.mat_exp_4d(A, t)

seq1 = seq_sim_cpp.seq_sim(seq0, P, seed = 2)

print(seq1)


H = div_mat_cpp.div_mat(seq0, seq1)
print(H)
