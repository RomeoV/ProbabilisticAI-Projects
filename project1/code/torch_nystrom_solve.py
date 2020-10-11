import math
import torch
from itertools import product

def nystrom_solve(X, t, m, sigma, k, compute_error=False):
    """ Solves (K + sigma I) a = t

    Williams, Christopher KI, and Matthias Seeger.
    "Using the Nyström method to speed up kernel machines."
    Advances in neural information processing systems. 2001.

    https://papers.nips.cc/paper/1866-using-the-nystrom-method-to-speed-up-kernel-machines.pdf

    """
    dtype = X.dtype

    # Randomly shuffle X
    n = X.shape[0]
    perm = torch.randperm(n)
    Xp = X[perm, :]

    # Compute reduced kernel matrix
    Knm = torch.zeros(n, m, dtype=dtype)
    for (i, j) in product(torch.arange(n), torch.arange(m)):
        Knm[i, j] = k(Xp[i, :], Xp[j, :])

    # Compute eigen-decomposition (see eq. (7))
    Lambda_m, U_m = torch.symeig(Knm[:m, :], eigenvectors=True)

    # Compute approximate eigenvalues (see eq. (8))
    Lambda = (n / m) * Lambda_m

    # Compute approximate eigenvectors (see eq. (9))
    U = math.sqrt(m / n) * Knm @ U_m @ torch.diag( 1.0 / Lambda_m )

    # Solve linear system (see eq. (11))
    Lambda = torch.diag(Lambda)
    y = Lambda @ U.T @ t
    z, lu_m = torch.solve(y.unsqueeze(1), Lambda @ U.T @ U + sigma * torch.eye(m))
    alpha = 1.0 / sigma * (t - U @ z)

    if compute_error:
        # Compute full kernel matrix
        Knn = torch.zeros(n, n, dtype=dtype)
        for (i, j) in product(torch.arange(n), torch.arange(n)):
            Knn[i, j] = k(Xp[i, :], Xp[j, :])
        alpha_true, lu_n = torch.solve(t.unsqueeze(1), Knn + sigma * torch.eye(n))
        print(torch.norm(alpha - alpha_true).item())

    return alpha

if __name__ == '__main__':
    dtype = torch.float64
    n = 1000
    X = torch.rand(n, 2, dtype=dtype)
    t = torch.ones(n, dtype=dtype)
    m = 100
    sigma = 1e-1
    k = lambda x, y : torch.exp(-torch.norm(x - y)**2).item()
    alpha = nystrom_solve(X, t, m, sigma, k, compute_error=True)
