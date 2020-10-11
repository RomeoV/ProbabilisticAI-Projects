import numpy as np
from itertools import product

def nystrom_solve(X, t, m, sigma, k, compute_error=False):
    """ Solves (K + sigma I) a = t

    Williams, Christopher KI, and Matthias Seeger.
    "Using the Nyström method to speed up kernel machines."
    Advances in neural information processing systems. 2001.

    https://papers.nips.cc/paper/1866-using-the-nystrom-method-to-speed-up-kernel-machines.pdf

    """

    # Randomly shuffle X
    n = X.shape[0]
    perm = np.random.permutation(n)
    Xp = X[perm, :]

    # Compute reduced kernel matrix
    Knm = np.zeros((n, m))
    for (i, j) in product(np.arange(n), np.arange(m)):
        Knm[i, j] = k(Xp[i, :], Xp[j, :])

    # Compute eigen-decomposition (see eq. (7))
    Lambda_m, U_m = np.linalg.eig(Knm[:m, :])

    # Compute approximate eigenvalues (see eq. (8))
    Lambda = (n / m) * Lambda_m

    # Compute approximate eigenvectors (see eq. (9))
    U = np.sqrt(m / n) * Knm @ U_m @ np.diag( 1.0 / Lambda_m )

    # Solve linear system (see eq. (11))
    Lambda = np.diag(Lambda)
    y = Lambda @ U.T @ t
    z = np.linalg.solve(Lambda @ U.T @ U + sigma * np.eye(m), y)
    alpha = 1 / sigma * (t - U @ z)

    if compute_error:
        # Compute full kernel matrix
        Knn = np.zeros((n, n))
        for (i, j) in product(np.arange(n), np.arange(n)):
            Knn[i, j] = k(Xp[i, :], Xp[j, :])
        alpha_true = np.linalg.solve(Knn + sigma * np.eye(n), t)
        print(np.linalg.norm(alpha - alpha_true))

    return alpha

if __name__ == '__main__':
    n = 1000
    X = np.random.rand(n, 2)
    t = np.ones(n)
    m = 100
    sigma = 1e-1
    k = lambda x, y : np.exp(-np.linalg.norm(x - y)**2)
    alpha = nystrom_solve(X, t, m, sigma, k, compute_error=True)
