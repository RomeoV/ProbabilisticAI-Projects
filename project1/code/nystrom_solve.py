import numpy as np
from itertools import product

def nystrom_solve(X, t, m, sigma, k, compute_error=False):
    """ Solves (K + sigma I) a = t

    Williams, Christopher KI, and Matthias Seeger.
    "Using the Nyström method to speed up kernel machines."
    Advances in neural information processing systems. 2001.

    https://papers.nips.cc/paper/1866-using-the-nystrom-method-to-speed-up-kernel-machines.pdf

    """

    n = X.shape[0]
    perm = np.random.permutation(n)
    Xp = X[perm, :]
    Xm = Xp[:m, :]

    # Compute reduced kernel matrix
    Kmn = np.zeros((m, n))
    for (i, j) in product(np.arange(m), np.arange(n)):
        Kmn[i, j] = k(Xm[i, :], Xp[j, :])
    Kmm = Kmn[:, :m]

    # Eq 7
    Lambda_m, U_m = np.linalg.eig(Kmm)

    # Eq 8
    Lambda = n / m * Lambda_m

    # Eq 9
    U = np.sqrt(m / n) * Kmn.T @ U_m @ np.diag( 1.0 / Lambda_m )

    # Eq 11
    Lambda = np.diag(Lambda)
    Im = np.eye(m)
    y = Lambda @ U.T @ t
    z = np.linalg.solve(sigma * Im + Lambda @ U.T @ U, y)
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
    alpha = nystrom_solve(X, t, m, sigma, k, compute_error=False)
