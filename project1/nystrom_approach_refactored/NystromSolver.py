import math
import torch
import gpytorch
import timeit
from itertools import product

class NystromSolver:
    """ Solves (K + sigma I) a = t using Nystrom approximation method.

    Williams, Christopher KI, and Matthias Seeger.
    "Using the Nyström method to speed up kernel machines."
    Advances in neural information processing systems. 2001.

    https://papers.nips.cc/paper/1866-using-the-nystrom-method-to-speed-up-kernel-machines.pdf

    """
    def __init__(self, X, k, m, sigma):
        self.X = X
        self.k = k
        self.m = m
        self.sigma = sigma
        self.dtype = X.dtype

    def preprocess(self):
        X = self.X
        k = self.k
        sigma = self.sigma
        n, m = X.shape[0], self.m

        # Compute reduced kernel matrix
        Knm = k(X[:n, :], X[:m, :]).evaluate()

        # Compute eigen-decomposition (see eq. (7))
        Lambda_m, U_m = torch.symeig(Knm[:m, :], eigenvectors=True)

        # Compute approximate eigenvalues (see eq. (8))
        Lambda = (n / m) * Lambda_m

        # Compute approximate eigenvectors (see eq. (9))
        U = math.sqrt(m / n) * Knm @ U_m @ torch.diag( 1.0 / Lambda_m )

        # Convert Lambda to diagonal matrix
        Lambda = torch.diag(Lambda)

        # Store approximate eigenvalues and eigenvectors
        self.Lambda, self.U = Lambda, U

        # Factorize M
        M = Lambda @ U.t() @ U + sigma * torch.eye(m)

        # Compute and store LU factors of M
        self.LU = torch.lu(M.detach())

    def solve(self, t):
        sigma = self.sigma
        m = self.m
        Lambda, U = self.Lambda, self.U
        LU = self.LU

        # Solve linear system (see eq. (11))
        y = Lambda @ U.t() @ t

        z = torch.lu_solve(y, *LU)

        alpha = 1.0 / sigma * (t - U @ z)

        return alpha


def main():
    torch.manual_seed(0)
    n = 1000                         # Number of total samples
    m = 100                          # Number of samples to be used in approximation
    k = gpytorch.kernels.RBFKernel() # Kernel function
    sigma = 1e-1                     # Damping parameter
    dtype = torch.float64            # Datatype

    # Generate dataset and setup solution of linear system
    X = torch.rand(n, 2, dtype=dtype)
    K = k(X, X).evaluate()
    I = torch.eye(n)
    a = torch.arange(n, dtype=dtype).unsqueeze(1) / n
    t = (K + sigma * I) @ a

    # Setup permutation
    perm = torch.randperm(n)

    # Setup inverse permutation
    iperm = torch.zeros(n, dtype=perm.dtype)
    iperm[perm] = torch.arange(n)

    # Nystrom approximation method
    solver = NystromSolver(X[perm, :], k, m, sigma)
    solver.preprocess()
    ahat = solver.solve(t[perm, :])
    ahat = ahat[iperm, :]

    # Print error of approximation
    relerr = torch.norm(a - ahat) / torch.norm(a)
    relres = torch.norm(t - (K + sigma * I) @ ahat) / torch.norm(t)
    print(f'Relative Error     = {relerr.item()}')
    print(f'Relative Residual  = {relres.item()}')


if __name__ == '__main__':
    main()
