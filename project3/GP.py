import torch

class GP():
    def __init__(self, mu, sigma, kernel, input_dim=1):
        self.mu = mu
        self.sigma = sigma
        self.kernel = kernel
        self.xs = torch.zeros(0, input_dim).double()
        self.ys = torch.zeros(0).double()
        self.K_sigma2_inv = torch.zeros(0, 0).double()
        self.num_points = 0


    @staticmethod
    def _get_blockwise_inverse(A_inv, B, C, D):
        """ Computes blockwise inverse using Schur complement """

        def _assemble_block_matrix(A, B, C, D):
            n = A.shape[0]
            M = torch.zeros(n+1, n+1).double()
            M[:n, :n] = A
            if n > 0:
                M[:n, n] = B.squeeze()
                M[n, :n] = C.squeeze()
            M[n, n] = D
            return M

        # compute Schur complement inverse
        S_A_inv = (D - C @ A_inv @ B).inverse()

        # compute blocks of inverse
        A_ = A_inv + A_inv @ B @ S_A_inv @ C @ A_inv
        B_ = -A_inv @ B @ S_A_inv
        C_ = B_.t() # symmetry
        D_ = S_A_inv

        # assemble blockwise inverse from its blocks
        M_inv = _assemble_block_matrix(A_, B_, C_, D_)

        return M_inv


    def add_point_and_update_kernel_inverse(self, xstar, ystar, niters=2):
        """ Adds point to GP and updates kernel inverse """
        # unpack
        sigma = self.sigma
        kernel = self.kernel
        xs = self.xs
        ys = self.ys

        xstar = torch.tensor(xstar)
        ystar = torch.tensor(ystar).unsqueeze(dim=0)

        # compute blockwise inverse
        A_inv = self.K_sigma2_inv
        B = kernel(xs, xstar)
        D = kernel(xstar, xstar) + sigma**2
        M_inv = self._get_blockwise_inverse(A_inv, B, B.t(), D)

        # do some iterative refinement steps
        n = M_inv.shape[0]
        I = torch.eye(n)
        xs = torch.cat((xs, xstar), dim=0)
        ys = torch.cat((ys, ystar), dim=0)
        M = kernel(xs, xs) + sigma**2 * I
        for i in range(niters):
            M_inv = M_inv @ (I + (I - M @ M_inv))

        # pack
        self.xs = xs
        self.ys = ys
        self.K_sigma2_inv = M_inv
        self.num_points += 1


    def predict(self, xstar):
        # unpack
        mu = self.mu
        kernel = self.kernel
        xA = self.xs
        yA = self.ys
        K_AA_sigma2_inv = self.K_sigma2_inv

        xstar = torch.tensor(xstar).unsqueeze(dim=0)

        # predict
        k_xA = kernel(xstar, xA)
        k_xx = kernel(xstar, xstar)

        mu_pred = mu + k_xA @ K_AA_sigma2_inv @ (yA - mu)

        sigma_pred = k_xx.diagonal() - (k_xA @ K_AA_sigma2_inv @ k_xA.t()).diagonal()

        return mu_pred, torch.sqrt(sigma_pred)


if __name__ == "__main__":
    from sklearn.gaussian_process.kernels import Matern
    kernel = Matern(length_scale=0.5, nu=2.5)
    myGP = GP(0.0, 0.15, lambda x, y: 0.5 * torch.from_numpy(kernel(x, y)))

