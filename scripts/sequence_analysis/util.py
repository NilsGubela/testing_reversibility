import numpy as np
from scipy.linalg import expm


NUCS = np.array(list("ACGT"))
IDX = {b:i for i,b in enumerate(NUCS)}

def random_seq(n, freqs=(0.25,0.25,0.25,0.25), seed = 1):
    np.random.seed(seed)
    freqs = np.asarray(freqs, dtype=float)
    freqs = freqs / freqs.sum()
    return "".join(np.random.choice(NUCS, size=n, p=freqs))

def evolve_discrete(seq, P, seed = 1):
    """
    seq: string of A/C/G/T
    P: 4x4 row-stochastic transition matrix; P[i,j]=Pr(i->j) in one step
    """
    np.random.seed(seed)
    P = np.asarray(P, dtype=float)
    x = np.fromiter((IDX[c] for c in seq), dtype=int)
    y = np.empty_like(x)
    for i in range(4):
        mask = (x == i)
        if mask.any():
            y[mask] = np.random.choice(4, size=mask.sum(), p=P[i])
    return "".join(NUCS[y])

def evolve_ctmc(seq, Q, t, seed = 1):
    """
    Q: 4x4 rate matrix (rows sum to 0, off-diagonals >=0)
    t: branch length / time
    """
    P = expm(np.asarray(Q, dtype=float) * t)
    return evolve_discrete(seq, P, seed = seed)


def sequence_diversity_matrix_two_sequences(seq1, seq2):
    """
    Sequence diversity matrix for TWO aligned sequences (same length).

    Output M is 4x4 where:
      M[i,j] = fraction of sites where seq1 has alphabet[i] and seq2 has alphabet[j].

    So M sums to 1. Row sums = composition of seq1, column sums = composition of seq2.

    Parameters
    ----------
    seq1, seq2 : str (or list/array of chars)
        Aligned sequences of equal length.

    Returns
    -------
    M : (4,4) ndarray of float
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must have the same length")

    M = np.zeros((4, 4), dtype=float)

    for a, b in zip(seq1, seq2):
        if a in IDX and b in IDX:          # ignore gaps/ambiguous chars
            M[IDX[a], IDX[b]] += 1.0

    return M



def get_d(P):
    """
    Python/Numpy translation of the Eigen C++ function get_d.

    Parameters
    ----------
    P : array-like, shape (4,4)

    Returns
    -------
    out : ndarray, shape (4,)
        [d1, d2, d3, d4]
    """
    P = np.asarray(P, dtype=float)
    if P.shape != (4, 4):
        raise ValueError("P must be a 4x4 matrix")

    d1 = P[0,1]*P[1,2]*P[2,0] - P[0,2]*P[2,1]*P[1,0]
    d2 = P[0,1]*P[1,3]*P[3,0] - P[0,3]*P[3,1]*P[1,0]
    d3 = P[0,2]*P[2,3]*P[3,0] - P[0,3]*P[3,2]*P[2,0]
    d4 = P[1,2]*P[2,3]*P[3,1] - P[1,3]*P[3,2]*P[2,1]

    return np.array([d1, d2, d3, d4], dtype=float)


def get_var(P, N, d1):
    """
    Python/Numpy translation of:

        double get_var(Eigen::Matrix<double, 4, 4> P, int N, int d1)

    Assumes you have Python versions of:
      - get_p(P, d1)            -> returns length-6 vector/array (p1..p6)
      - get_expect_sqrt1(N, a,b,c)

    Parameters
    ----------
    P : array-like, shape (4,4)
    N : int
    d1 : int

    Returns
    -------
    out : float
    """
    P = np.asarray(P, dtype=float)
    tmp_p = get_p(P, d1)              # should be shape (6,) or (6,1)
    tmp_p = np.asarray(tmp_p, dtype=float).reshape(-1)
    p1, p2, p3, p4, p5, p6 = tmp_p[:6]

    # tmp = E[a - b]
    tmp = (
        np.exp(np.log(N) + np.log(N-1) + np.log(N-2) + np.log(p1) + np.log(p2) + np.log(p3))
        - np.exp(np.log(N) + np.log(N-1) + np.log(N-2) + np.log(p4) + np.log(p5) + np.log(p6))
    )

    # tmp1 = E[a^2]
    tmp1 = get_expect_sqrt1(N, p1, p2, p3)

    # tmp2 = E[ab]
    tmp2 = np.exp(
        np.log(N) + np.log(N-1) + np.log(N-2) + np.log(N-3) + np.log(N-4) + np.log(N-5)
        + np.log(p1) + np.log(p2) + np.log(p3) + np.log(p4) + np.log(p5) + np.log(p6)
    )

    # tmp3 = E[b^2]
    tmp3 = get_expect_sqrt1(N, p4, p5, p6)

    out = tmp1 - 2.0 * tmp2 + tmp3 - tmp * tmp
    return float(out)

def get_expect_sqrt1(N, p1, p2, p3):
    """
    Python translation of:

    double get_expect_sqrt1(int N, double p1, double p2, double p3)

    Parameters
    ----------
    N : int
    p1, p2, p3 : float

    Returns
    -------
    float
    """
    tmp  = np.log(p1*p2 + p1*p3 + p2*p3)
    tmp2 = np.log(p1*p2*p3)

    out = np.log(
        1
        + (N - 3) * (p1 + p2 + p3)
        + np.exp(np.log(N - 3) + np.log(N - 4) + tmp)
        + np.exp(np.log(N - 3) + np.log(N - 4) + np.log(N - 5) + tmp2)
    )

    return float(np.exp(np.log(N) + np.log(N - 1) + np.log(N - 2) + tmp2 + out))


def get_p(P, d1):
    """
    Python/Numpy translation of the Eigen function get_p.

    Parameters
    ----------
    P : array-like, shape (4,4)
    d1 : int
        Must be 1, 2, 3, or 4.

    Returns
    -------
    out : ndarray, shape (6,)
        The selected 6 entries of P in the same order as the C++ code.
    """
    P = np.asarray(P, dtype=float)
    if P.shape != (4, 4):
        raise ValueError("P must be a 4x4 matrix")

    if d1 == 1:
        out = np.array([P[0,1], P[1,2], P[2,0], P[0,2], P[2,1], P[1,0]], dtype=float)
    elif d1 == 2:
        out = np.array([P[0,1], P[1,3], P[3,0], P[0,3], P[3,1], P[1,0]], dtype=float)
    elif d1 == 3:
        out = np.array([P[0,2], P[2,3], P[3,0], P[0,3], P[3,2], P[2,0]], dtype=float)
    elif d1 == 4:
        out = np.array([P[1,2], P[2,3], P[3,1], P[1,3], P[3,2], P[2,1]], dtype=float)
    else:
        raise ValueError("d1 must be 1, 2, 3, or 4")

    return out




def get_m(H):
    H = np.asarray(H)
    out = np.empty(6, dtype=float)
    k = 0
    for i in range(4):
        for j in range(i + 1, 4):
            out[k] = abs(H[i, j]) - abs(H[j, i])
            k += 1
    return out

def get_B(H):
    H = np.asarray(H)
    out = np.zeros((6, 6), dtype=float)
    k = 0
    for i in range(4):
        for j in range(4):
            if i < j:
                out[k, k] = H[i, j] + H[j, i]
                k += 1
    return out

def bowker_stat(m, B):
    m = np.asarray(m, dtype=float).reshape(6, 1)
    B = np.asarray(B, dtype=float)

    #stat = float(m.T @ np.linalg.inv(B) @ m)
    stat = (m.T @ np.linalg.solve(B, m)).item()
    return stat


def stuart_from_counts(N):
    N = np.asarray(N)
    if N.ndim != 2 or N.shape[0] != N.shape[1]:
        raise ValueError("counts must be a square 2D array.")
    K = N.shape[0]
    if K < 2:
        return 0.0 # stat, df, rank
    r = N.sum(axis=1)
    c = N.sum(axis=0)
    d_full = (r - c).astype(float)

    m = K - 1
    d = d_full[:m]

    V = np.zeros((m, m), dtype=float)
    for i in range(m):
        V[i, i] = r[i] + c[i] - 2 * N[i, i]
        for j in range(m):
            if i != j:
                V[i, j] = -(N[i, j] + N[j, i])

    rank = int(np.linalg.matrix_rank(V))
    if rank == 0:
        return 0.0
    stat = float(d.T @ np.linalg.pinv(V) @ d)
    return stat


def qs_test(H, n):
    # Proposed test for quasi-symmetry
    P_hat = H / n

    d = get_d(H)

    V2 = np.zeros((4, 4))

    # Sum of variances
    var = 4.0
    stat = 0.0

    # Construct covariance matrix
    for i in range(4):
        for j in range(i, 4):
            if i == j:
                V2[i, j] = get_var(P_hat, n, i + 1)
            else:
                V2[i, j] = get_covar(P_hat,n,i + 1,j + 1)
                V2[j, i] = V2[i, j]

    # Compute test statistic
    for i in range(4):
        stat += d[i] / np.sqrt(V2[i, i])
        for j in range(4):
            if i != j:
                var += (1 / np.sqrt(V2[i, i])* 1 / np.sqrt(V2[j, j])* V2[i, j])

    test_statistic = stat / np.sqrt(var)

    return(test_statistic)

import numpy as np


def get_covar_help(P, N, d1, d2):

    tmp_p = get_p(P, d1)

    p1, p2, p3, p4, p5, p6 = tmp_p.flatten()

    tmp_p = get_p(P, d2)

    b1, b2, b3, b4, b5, b6 = tmp_p.flatten()

    if d1 == 1:

        if d2 == 2:

            tmp1 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4)
                + np.log(p2) + np.log(p3)
                + np.log(b1) + np.log(b2) + np.log(b3)
            ) * (1 + (N-5)*p1)

            tmp2 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4) + np.log(N-5)
                + np.log(p1) + np.log(p2) + np.log(p3)
                + np.log(b4) + np.log(b5) + np.log(b6)
            )

            tmp3 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4) + np.log(N-5)
                + np.log(p4) + np.log(p5) + np.log(p6)
                + np.log(b1) + np.log(b2) + np.log(b3)
            )

            tmp4 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4)
                + np.log(p4) + np.log(p5)
                + np.log(b4) + np.log(b5) + np.log(b6)
            ) * (1 + (N-5)*p6)

        elif d2 == 3:

            tmp1 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4) + np.log(N-5)
                + np.log(p1) + np.log(p2) + np.log(p3)
                + np.log(b1) + np.log(b2) + np.log(b3)
            )

            tmp2 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4)
                + np.log(p1) + np.log(p2)
                + np.log(b4) + np.log(b5) + np.log(b6)
            ) * (1 + (N-5)*p3)

            tmp3 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4)
                + np.log(p5) + np.log(p6)
                + np.log(b1) + np.log(b2) + np.log(b3)
            ) * (1 + (N-5)*p4)

            tmp4 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4) + np.log(N-5)
                + np.log(p4) + np.log(p5) + np.log(p6)
                + np.log(b4) + np.log(b5) + np.log(b6)
            )

        elif d2 == 4:

            tmp1 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4)
                + np.log(p1) + np.log(p3)
                + np.log(b1) + np.log(b2) + np.log(b3)
            ) * (1 + (N-5)*p2)

            tmp2 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4) + np.log(N-5)
                + np.log(p1) + np.log(p2) + np.log(p3)
                + np.log(b4) + np.log(b5) + np.log(b6)
            )

            tmp3 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4) + np.log(N-5)
                + np.log(p4) + np.log(p5) + np.log(p6)
                + np.log(b1) + np.log(b2) + np.log(b3)
            )

            tmp4 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4)
                + np.log(p4) + np.log(p6)
                + np.log(b4) + np.log(b5) + np.log(b6)
            ) * (1 + (N-5)*p5)

    elif d1 == 2:

        if d2 == 3:

            tmp1 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4)
                + np.log(p1) + np.log(p2)
                + np.log(b1) + np.log(b2) + np.log(b3)
            ) * (1 + (N-5)*p3)

            tmp2 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4) + np.log(N-5)
                + np.log(p1) + np.log(p2) + np.log(p3)
                + np.log(b4) + np.log(b5) + np.log(b6)
            )

            tmp3 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4) + np.log(N-5)
                + np.log(p4) + np.log(p5) + np.log(p6)
                + np.log(b1) + np.log(b2) + np.log(b3)
            )

            tmp4 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4)
                + np.log(p5) + np.log(p6)
                + np.log(b4) + np.log(b5) + np.log(b6)
            ) * (1 + (N-5)*p4)

        elif d2 == 4:

            tmp1 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4) + np.log(N-5)
                + np.log(p1) + np.log(p2) + np.log(p3)
                + np.log(b1) + np.log(b2) + np.log(b3)
            )

            tmp2 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4)
                + np.log(p1) + np.log(p3)
                + np.log(b4) + np.log(b5) + np.log(b6)
            ) * (1 + (N-5)*p2)

            tmp3 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4)
                + np.log(p4) + np.log(p6)
                + np.log(b1) + np.log(b2) + np.log(b3)
            ) * (1 + (N-5)*p5)

            tmp4 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4) + np.log(N-5)
                + np.log(p4) + np.log(p5) + np.log(p6)
                + np.log(b4) + np.log(b5) + np.log(b6)
            )

    elif d1 == 3:

        if d2 == 4:

            tmp1 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4)
                + np.log(p1) + np.log(p3)
                + np.log(b1) + np.log(b2) + np.log(b3)
            ) * (1 + (N-5)*p2)

            tmp2 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4) + np.log(N-5)
                + np.log(p1) + np.log(p2) + np.log(p3)
                + np.log(b4) + np.log(b5) + np.log(b6)
            )

            tmp3 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4) + np.log(N-5)
                + np.log(p4) + np.log(p5) + np.log(p6)
                + np.log(b1) + np.log(b2) + np.log(b3)
            )

            tmp4 = np.exp(
                np.log(N) + np.log(N-1) + np.log(N-2)
                + np.log(N-3) + np.log(N-4)
                + np.log(p4) + np.log(p6)
                + np.log(b4) + np.log(b5) + np.log(b6)
            ) * (1 + (N-5)*p5)

    out = tmp1 - tmp2 - tmp3 + tmp4

    return out


def get_covar(P, N, d1, d2):

    tmp_p = get_p(P, d1)

    p1, p2, p3, p4, p5, p6 = tmp_p.flatten()

    exp_d1 = (
        np.exp(
            np.log(N) + np.log(N-1) + np.log(N-2)
            + np.log(p1) + np.log(p2) + np.log(p3)
        )
        -
        np.exp(
            np.log(N) + np.log(N-1) + np.log(N-2)
            + np.log(p4) + np.log(p5) + np.log(p6)
        )
    )

    tmp_p = get_p(P, d2)

    b1, b2, b3, b4, b5, b6 = tmp_p.flatten()

    exp_d2 = (
        np.exp(
            np.log(N) + np.log(N-1) + np.log(N-2)
            + np.log(b1) + np.log(b2) + np.log(b3)
        )
        -
        np.exp(
            np.log(N) + np.log(N-1) + np.log(N-2)
            + np.log(b4) + np.log(b5) + np.log(b6)
        )
    )

    out = (
        get_covar_help(P, N, d1, d2)
        - exp_d1 * exp_d2
    )

    return out


def get_Dk(H, k=1):

    diag = H[k, :] / H[:, k]

    return np.diag(diag)

def get_fDN(D, H):

    out = D @ H

    offdiag = out.copy()

    np.fill_diagonal(offdiag, 0)

    row_sum = offdiag.sum(axis=1)

    col_sum = offdiag.sum(axis=0)

    D_new = np.diag(row_sum / col_sum)

    return out @ D_new

def stat_distP(A):

    n = A.shape[0]

    M = A.T - np.eye(n)

    M[-1] = np.ones(n)

    b = np.zeros(n)

    b[-1] = 1

    pi = np.linalg.solve(M, b)

    return pi.reshape((n, 1))

def saturation(H):
    n = np.sum(H)
    pi_star = (np.sum(H,axis = 0) + np.sum(H, axis = 1))/(2*n)
    res = -1
    for i in range(4):
        res += H[i,i]/n * 1/pi_star[i]
    return(res)




