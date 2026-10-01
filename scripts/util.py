import numpy as np
import numpy as np
from scipy.linalg import expm


def random_reversible_Q4(seed=1):
    """
    Random 4x4 reversible rate matrix with mean rate 1.

    Steps:
      1) pi ~ Dirichlet(1,1,1,1)
      2) draw 6 exchangeabilities for pairs (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) ~ U(0,1)
      3) build symmetric S with zeros on diagonal
      4) set q_ij = s_ij * pi_j  (i != j)  -> ensures detailed balance
      5) q_ii = -sum_{j!=i} q_ij
      6) scale so -sum_i pi_i q_ii = 1

    Returns
    -------
    Q : (4,4) ndarray
    pi : (4,) ndarray
    S : (4,4) ndarray (symmetric exchangeabilities)
    """
    
    np.random.seed(seed) 

    # 1) stationary distribution
    pi = np.random.dirichlet(np.ones(4))

    # 2) six random exchangeabilities
    r = np.random.uniform(0.0, 1.0, size=6)

    # 3) symmetric S (diag=0)
    S = np.zeros((4, 4), dtype=float)
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for k, (i, j) in enumerate(pairs):
        S[i, j] = r[k]
        S[j, i] = r[k]

    # 4) build Q off-diagonals: q_ij = s_ij * pi_j
    Q = S * pi[np.newaxis, :]  # multiply each column j by pi_j

    # 5) diagonals
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))

    # 6) normalize to mean rate 1
    avg_rate = -np.dot(pi, np.diag(Q))  # = sum_i pi_i * (-q_ii)
    Q /= avg_rate

    return Q, pi, S


def random_nonreversible_Q4(seed=1):
    """
    Random 4x4 reversible rate matrix with mean rate 1.

    Steps:
      1) pi ~ Dirichlet(1,1,1,1)
      2) draw 6 exchangeabilities for pairs ((0,1),(2,3)), ((0,2),(2,0)), ((0,3),(2,1)), ((1,0),(3,2)), ((1,2),(3,0)), ((1,3), (3,1)) ~ U(0,1)
      3) build symmetric S with zeros on diagonal
      4) set q_ij = s_ij * pi_j  (i != j)  -> ensures detailed balance
      5) q_ii = -sum_{j!=i} q_ij
      6) scale so -sum_i pi_i q_ii = 1

    Returns
    -------
    Q : (4,4) ndarray
    pi : (4,) ndarray
    """
    
    np.random.seed(seed) 

    # 1) stationary distribution
    pi = np.random.dirichlet(np.ones(4))

    # 2) six random exchangeabilities
    r = np.random.uniform(0.0, 1.0, size=6)

    # 3) A (diag=0)
    A = np.zeros((4, 4), dtype=float)
    pair1 = [(0,1), (0,2), (0,3), (1,0), (1,2), (1,3)]
    pair2 = [(2,3), (2,0), (2,1), (3,2), (3,0), (3,1)]
    for k in range(len(pair1)):
        A[pair1[k]] = r[k]
        A[pair2[k]] = r[k]

    # 4) build Q off-diagonals: q_ij = s_ij * pi_j
    Q = A * pi[np.newaxis, :]  # multiply each column j by pi_j

    # 5) diagonals
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))

    # 6) normalize to mean rate 1
    avg_rate = -np.dot(pi, np.diag(Q))  # = sum_i pi_i * (-q_ii)
    Q /= avg_rate

    return Q, pi


def check_detailed_balance(Q, pi, atol=1e-10):
    Q = np.asarray(Q, float)
    pi = np.asarray(pi, float)
    K = Q.shape[0]
    M = pi[:, None] * Q
    return np.max(np.abs(M - M.T) * (1 - np.eye(K))), np.allclose(M, M.T, atol=atol)

def return_detailed_balance(Q, pi):
    Q = np.asarray(Q, dtype=float)
    pi = np.asarray(pi, dtype=float).reshape(-1)

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q must be square, got shape {Q.shape}")

    K = Q.shape[0]

    if pi.shape[0] != K:
        raise ValueError(f"pi must have length {K}, got shape {pi.shape}")

    M = pi[:, None] * Q
    return float(np.sum(np.abs(M - M.T)))


NUCS = np.array(list("ACGT"))
IDX = {b:i for i,b in enumerate(NUCS)}

def random_seq(n, freqs=(0.25,0.25,0.25,0.25), seed = 1):
    np.random.seed(seed)
    freqs = np.asarray(freqs, dtype=float)
    freqs = freqs / freqs.sum()
    return "".join(np.random.choice(NUCS, size=n, p=freqs))

def evolve_discrete(seq, P, seed = 1):
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
    P = expm(np.asarray(Q, dtype=float) * t)
    return evolve_discrete(seq, P, seed = seed)




def sequence_diversity_matrix_two_sequences(seq1, seq2):

    if len(seq1) != len(seq2):
        raise ValueError("Sequences must have the same length")

    M = np.zeros((4, 4), dtype=float)

    for a, b in zip(seq1, seq2):
        if a in IDX and b in IDX:          # ignore gaps/ambiguous chars
            M[IDX[a], IDX[b]] += 1.0

    return M



def get_d(P):

    P = np.asarray(P, dtype=float)
    if P.shape != (4, 4):
        raise ValueError("P must be a 4x4 matrix")

    d1 = P[0,1]*P[1,2]*P[2,0] - P[0,2]*P[2,1]*P[1,0]
    d2 = P[0,1]*P[1,3]*P[3,0] - P[0,3]*P[3,1]*P[1,0]
    d3 = P[0,2]*P[2,3]*P[3,0] - P[0,3]*P[3,2]*P[2,0]
    d4 = P[1,2]*P[2,3]*P[3,1] - P[1,3]*P[3,2]*P[2,1]

    return np.array([d1, d2, d3, d4], dtype=float)

def _safe_prod_log(*xs):

    xs = np.asarray(xs, dtype=float)

    if np.any(xs < 0):
        raise ValueError("All factors must be nonnegative.")

    if np.any(xs == 0):
        return 0.0

    return float(np.exp(np.sum(np.log(xs))))


def get_var(P, N, d1):
    P = np.asarray(P, dtype=float)
    tmp_p = get_p(P, d1)
    tmp_p = np.asarray(tmp_p, dtype=float).reshape(-1)
    p1, p2, p3, p4, p5, p6 = tmp_p[:6]

    # tmp = E[a - b]
    term_a = _safe_prod_log(N, N-1, N-2, p1, p2, p3)
    term_b = _safe_prod_log(N, N-1, N-2, p4, p5, p6)
    tmp = term_a - term_b

    # tmp1 = E[a^2]
    tmp1 = get_expect_sqrt1(N, p1, p2, p3)

    # tmp2 = E[ab]
    tmp2 = _safe_prod_log(N, N-1, N-2, N-3, N-4, N-5, p1, p2, p3, p4, p5, p6)

    # tmp3 = E[b^2]
    tmp3 = get_expect_sqrt1(N, p4, p5, p6)

    out = tmp1 - 2.0 * tmp2 + tmp3 - tmp * tmp
    return float(out)


def get_var_old(P, N, d1):
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

def get_expect_sqrt1_old(N, p1, p2, p3):
    tmp  = np.log(p1*p2 + p1*p3 + p2*p3)
    tmp2 = np.log(p1*p2*p3)

    out = np.log(
        1
        + (N - 3) * (p1 + p2 + p3)
        + np.exp(np.log(N - 3) + np.log(N - 4) + tmp)
        + np.exp(np.log(N - 3) + np.log(N - 4) + np.log(N - 5) + tmp2)
    )

    return float(np.exp(np.log(N) + np.log(N - 1) + np.log(N - 2) + tmp2 + out))

def _logsumexp(log_terms):
    """
    Stable log(sum(exp(log_terms))) for finite / -inf log_terms.
    """
    log_terms = np.asarray(log_terms, dtype=float)

    m = np.max(log_terms)
    if np.isneginf(m):
        return -np.inf

    return float(m + np.log(np.sum(np.exp(log_terms - m))))

def get_expect_sqrt1(N, p1, p2, p3):
    p1 = float(p1)
    p2 = float(p2)
    p3 = float(p3)
    N = float(N)

    if min(p1, p2, p3) < 0:
        raise ValueError("Probabilities must be nonnegative.")

    if p1 == 0 or p2 == 0 or p3 == 0:
        return 0.0

    pair_sum = p1*p2 + p1*p3 + p2*p3
    triple_prod = p1*p2*p3

    log_terms = [
        0.0,
        np.log(N - 3) + np.log(p1 + p2 + p3),
        np.log(N - 3) + np.log(N - 4) + np.log(pair_sum),
        np.log(N - 3) + np.log(N - 4) + np.log(N - 5) + np.log(triple_prod),
    ]

    out_log = _logsumexp(log_terms)

    total_log = (
        np.log(N) + np.log(N - 1) + np.log(N - 2)
        + np.log(triple_prod)
        + out_log
    )

    max_log = np.log(np.finfo(float).max)
    if total_log > max_log:
        return np.inf

    return float(np.exp(total_log))


def get_p(P, d1):
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

def bowker_stat2(M):
    M = np.asarray(M, dtype=float)

    if M.shape != (4, 4):
        raise ValueError("Input must be a 4x4 matrix.")

    stat = 0.0
    df = 0

    for i in range(4):
        for j in range(i + 1, 4):
            a = M[i, j]
            b = M[j, i]
            s = a + b
            if s > 0:
                stat += (a - b) ** 2 / s
                df += 1

    return stat, df


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
        return np.nan
    stat = float(d.T @ np.linalg.pinv(V) @ d)
    return stat


def qs_test(H, n, size = 4):
    # Proposed test for quasi-symmetry
    P_hat = H / n

    d = get_d(H)

    V2 = np.zeros((size, size))

    # Sum of variances
    var = size
    stat = 0.0

    # Construct covariance matrix
    for i in range(size):
        for j in range(i, size):
            if i == j:
                V2[i, j] = get_var(P_hat, n, i + 1)
            else:
                V2[i, j] = get_covar(P_hat,n,i + 1,j + 1)
                V2[j, i] = V2[i, j]

    # Compute test statistic
    for i in range(size):
        stat += d[i] / np.sqrt(V2[i, i])
        for j in range(size):
            if i != j:
                var += (1 / np.sqrt(V2[i, i])* 1 / np.sqrt(V2[j, j])* V2[i, j])

    test_statistic = stat / np.sqrt(var)

    return(test_statistic)


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

def safe_standardize(x, var):
    var = float(var)
    x = float(x)

    if var < 0:
        # Variance should not be negative, but tiny negative values can happen numerically
        if np.isclose(var, 0.0):
            var = 0.0
        else:
            raise ValueError(f"Negative variance encountered: {var}")

    if var == 0.0:
        if x == 0.0:
            return np.nan
        return np.inf * np.sign(x)

    return x / np.sqrt(var)

