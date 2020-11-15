import numpy as np
from math import sqrt
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process.kernels import Matern
import torch
from typing import Tuple

domain = np.array([[0, 5]])
domain_t = torch.tensor(domain)


""" Solution """


class BO_algo:
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # GP parameters for f
        self.Matern_f_np = Matern(length_scale=0.5, nu=2.5)
        self.Matern_f = lambda x, y: self.var_f * torch.from_numpy(self.Matern_f_np(x, y))
        self.μf_prior = 0.0
        self.var_f = 0.5  # variance
        self.σ_f = 0.15  # measurement noise

        # GP parameters for v
        self.Matern_v_np = Matern(length_scale=0.5, nu=2.5)
        self.Matern_v = lambda x, y: self.var_v * torch.from_numpy(self.Matern_v_np(x, y))
        self.μv_prior = 1.5
        self.var_v = sqrt(2)
        self.σ_v = 0.0001

        self.κ = 1.2  # v_min
        self.β = 2.

        self.xs = torch.zeros(0, domain_t.shape[0]).double()
        self.fs = torch.zeros(0).double()
        self.vs = torch.zeros(0).double()

        self.Kf_AA_sig2_inv = torch.zeros(0, 0).double()
        self.Kv_AA_sig2_inv = torch.zeros(0, 0).double()

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        if (self.Kf_AA_sig2_inv.numel() == 0):
            retval = torch.tensor(domain).double().mean().unsqueeze(0).numpy()
            retval = 1.0  # TODO change this
        else:
            retval = self.optimize_acquisition_function()
        return np.atleast_2d(retval)

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        x = torch.tensor(x).unsqueeze(dim=0)
        Ɛ = 0.9
        β = 1.
        β2 = 1.

        k_xAf = self.Matern_f(x, self.xs)
        μf_pred = self.μf_prior + k_xAf @ self.Kf_AA_sig2_inv @ (self.fs - self.μf_prior)
        σf_pred = torch.sqrt(self.Matern_f(x, x) - k_xAf @ self.Kf_AA_sig2_inv @ k_xAf.t())

        k_xAv = self.Matern_v(x, self.xs)
        μv_pred = 1.5
        σv_pred = torch.sqrt(self.Matern_v(x, x) - k_xAv @ self.Kv_AA_sig2_inv @ k_xAv.t())
        if(μv_pred-σv_pred <= self.κ):
            return Ɛ*(μf_pred + β*σf_pred).item()+(1-Ɛ)*(μv_pred + β2*σv_pred).item()
        else:
            return (μf_pred + β*σf_pred).item()

    def get_mu_sigma(self, xs: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k_xA = self.Matern_f(xs, self.xs)
        μf_pred = self.μf_prior + k_xA @ self.Kf_AA_sig2_inv @ (self.fs - self.μf_prior)
        σf_pred = torch.sqrt(self.Matern_f(xs, xs).diagonal() - (k_xA @ self.Kf_AA_sig2_inv @ k_xA.t()).diagonal())
        return μf_pred, σf_pred

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here

        x = torch.tensor(x)
        f = torch.tensor(f).unsqueeze(dim=0)
        v = torch.tensor(v).unsqueeze(dim=0)

        # First for f
        assert(x.shape[0] == 1)
        Af_inv = self.Kf_AA_sig2_inv
        Bf = self.Matern_f(self.xs, x)
        Cf = Bf.t()
        Df = self.Matern_f(x, x) + self.σ_f**2

        Mf_inv = self._get_blockwise_inverse(Af_inv, Bf, Cf, Df)

        # First for v
        Av_inv = self.Kv_AA_sig2_inv
        Bv = self.Matern_v(self.xs, x)
        Cv = Bv.t()
        Dv = self.Matern_v(x, x) + self.σ_v**2

        Mv_inv = self._get_blockwise_inverse(Av_inv, Bv, Cv, Dv)

        # Append new values to data buffer
        self.xs = torch.cat((self.xs, x), dim=0)
        self.fs = torch.cat((self.fs, f), dim=0)
        self.vs = torch.cat((self.vs, v), dim=0)

        Mf = self.Matern_f(self.xs, self.xs) + self.σ_f**2 * torch.eye(self.xs.shape[0])
        Mv = self.Matern_v(self.xs, self.xs) + self.σ_v**2 * torch.eye(self.xs.shape[0])
        n = Mf.shape[0]
        if n >= 1:
            self.Kf_AA_sig2_inv = Mf_inv @ (torch.eye(n) + (torch.eye(n) - Mf @ Mf_inv))
            self.Kv_AA_sig2_inv = Mv_inv @ (torch.eye(n) + (torch.eye(n) - Mv @ Mv_inv))
        else:
            self.Kf_AA_sig2_inv = Mf_inv
            self.Kv_AA_sig2_inv = Mv_inv

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        candidates = self.xs[self.vs > self.κ]
        indices = np.where(candidates)[0]
        assert candidates.numel() > 0,\
            "No candidates satisfies speed constraint"

        return candidates[self.fs[indices].argmax()]

    @ staticmethod
    def _get_blockwise_inverse(A_inv, B, C, D):
        """ Efficient matrix inversion using Schur Complement

        Given M = [A, B ; C D] and A^{-1}
        then M^{-1} = [A_, B_; C_, D_] where
        A_ = A^{-1} + A^{-1} B S_A C A^{-1}
        B_ = -A^{-1} B S_A
        C_ = -S_A C A^{-1}, or just B_^T if M symmetric
        D_ = S_A
        and S_A = D - C A^{-1} B.

        For us, A = K_AA + \sigma^2 I
        B = k(xs, x_new), C = B_
        D = k(x_new, x_new) + \sigma^2
        """

        def _assemble_block_matrix(A, B, C, D):
            N = A.shape[0]
            M = torch.zeros(N+1, N+1).double()
            M[:N, :N] = A
            if N > 0:
                M[:N, N] = B.squeeze()
                M[N, :N] = C.squeeze()
            M[N, N] = D
            return M

        S_A_inv = (D - C @ A_inv @ B).inverse()

        A_ = A_inv + A_inv @ B @ S_A_inv @ C @ A_inv
        B_ = -A_inv @ B @ S_A_inv
        C_ = B_.t()
        D_ = S_A_inv

        M_inv = _assemble_block_matrix(A_, B_, C_, D_)

        return M_inv


""" Toy problem to check code works as expected """


def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return domain[:, 0] + np.random.rand(domain.shape[0]) * (domain[:, 1] - domain[:, 0])


def train_agent(agent, n_iters=20, debug=False):

   # Loop until budget is exhausted
    for j in range(n_iters):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)
        if debug:
            M1 = agent.Kf_AA_sig2_inv
            xs = agent.xs
            M2 = (agent.Matern_f(xs, xs) + agent.σ_f**2 * torch.eye(xs.shape[0])).inverse()
            print((M1 - M2).abs().sum())

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


def plot_agent(agent, ax=None):
    try:
        import matplotlib.pyplot as plt
        if ax == None:
            fig, ax = plt.subplots()
        xs = torch.linspace(0, 5)

        ys = np.array(list(map(f, xs)))
        mus, sigs = agent.get_mu_sigma(xs.unsqueeze(1))
        ax.plot(xs, ys, label="GT")
        ax.plot(xs, mus, '--', label="Mean")
        ax.plot(xs, mus+sigs, '-.', c='g', label="Mean + std")
        ax.plot(xs, mus-sigs, '-.', c='g', label="Mean - std")
        ax.scatter(agent.xs, agent.fs, label="Sample points")
        ax.legend()
    except ImportError:
        pass


def main():
    # Init problem
    agent = BO_algo()
    train_agent(agent, debug=True)
    plot_agent(agent)


if __name__ == "__main__":
    main()
