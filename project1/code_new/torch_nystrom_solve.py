import math
import torch
from itertools import product
import gpytorch as gpt

class NystromSolverThingVeryFast:
    def nystrom_decomp(self, X, m, sigma, kernel):
        """ Solves (K + sigma I) a = t

        Williams, Christopher KI, and Matthias Seeger.
        "Using the Nyström method to speed up kernel machines."
        Advances in neural information processing systems. 2001.

        https://papers.nips.cc/paper/1866-using-the-nystrom-method-to-speed-up-kernel-machines.pdf

        """
        self.sigma = sigma
        n = X.shape[0]

        # Compute reduced kernel matrix
        Knm = kernel(X[:n], X[:m]).evaluate()

        # Compute eigen-decomposition (see eq. (7))
        Lambda_m, U_m = torch.symeig(Knm[:m, :], eigenvectors=True)

        # Compute approximate eigenvalues (see eq. (8))
        Lambda = (n / m) * Lambda_m

        # Compute approximate eigenvectors (see eq. (9))
        self.U = math.sqrt(m / n) * Knm @ U_m @ torch.diag( 1.0 / Lambda_m )

        # Solve linear system (see eq. (11))
        self.Lambda = torch.diag(Lambda)
        self.LU, self.pivots = torch.lu(self.Lambda @ self.U.T @ self.U + self.sigma * torch.eye(m))

    def nystrom_solve(self, rhs):
        y = self.Lambda @ self.U.T @ rhs
        print(f"Estimated nystrom matrix size: {y.shape[0]**2 * 8 / 1000000} MB")
        z = torch.lu_solve(y, self.LU, self.pivots)
        alpha = 1.0 / self.sigma * (rhs - (self.U @ z))
        return alpha

    def nystrom_solve_old_af(X, t, m, sigma, k, compute_error=False, use_gpytorch_kernel=True):
        """ Solves (K + sigma I) a = t

        Williams, Christopher KI, and Matthias Seeger.
        "Using the Nyström method to speed up kernel machines."
        Advances in neural information processing systems. 2001.

        https://papers.nips.cc/paper/1866-using-the-nystrom-method-to-speed-up-kernel-machines.pdf

        """
        assert self.LU is not None, "Do nystrom_decomp first you idiot!"

        dtype = X.dtype

        n = X.shape[0]
        # perm = torch.arange(n)
        # Xp = X[perm, :]
        Xp = X

        ind1_mask = X[:,0] > -0.5
        ind2_mask = X[:,0] <= -0.5

        ind1 = torch.arange(n)[ind1_mask]
        ind1 = ind1[:min(m, ind1.shape[0])]

        #ind2 = torch.arange(m - ind1.shape[0])
        ind2 = torch.arange(n)[ind2_mask]
        ind = torch.cat((ind1, ind2))
        ind.sort()



        # Compute reduced kernel matrix
        # import IPython; IPython.embed(); exit()
        if use_gpytorch_kernel:
            Knm = k(Xp[:n], Xp[:m]).evaluate()
            #Knm = k(Xp[:n], Xp[ind]).evaluate()
        else:
            raise "not up to date anymore"
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
        dim = y.shape[0]
        print(f"Estimated nystrom matrix size: {dim**2 * 8 / 1000000} MB")
        LU, pivots, info = torch.lu(Lambda @ U.T @ U + sigma * torch.eye(m), get_infos=True)
        y.detach()
        LU.detach()
        z = torch.lu_solve(y, LU, pivots)
        #z, LU = torch.lu_solve(y, Lambda @ U.T @ U + sigma * torch.eye(m))
        alpha = 1.0 / sigma * (t - (U @ z))
        #alpha = 1.0 / sigma * (t - (U @ z))

        if compute_error:
            # Compute full kernel matrix
            if use_gpytorch_kernel:
                Knn = k(Xp[:n], Xp[:n]).evaluate()
            else:
                Knn = torch.zeros(n, n, dtype=dtype)
                for (i, j) in product(torch.arange(n), torch.arange(n)):
                    Knn[i, j] = k(Xp[i, :], Xp[j, :])
            alpha_true, _ = torch.solve(t, Knn + sigma * torch.eye(n))
            print(torch.norm(alpha - alpha_true).item())

        return (alpha, (LU, pivots))

    if __name__ == '__main__':
        dtype = torch.float64
        n = 100
        X = torch.rand(n, 2, dtype=dtype)
        t = torch.ones(n, 1, dtype=dtype)
        m = 100
        sigma = 1e-1
        k = lambda x, y : torch.exp(-torch.norm(x - y)**2).item()
        #k = gpt.kernels.RBFKernel()
        alpha = nystrom_solve(X, t, m, sigma, k, compute_error=True, use_gpytorch_kernel=False)
