import torch
import math
import numpy as np
from math import sqrt
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process.kernels import Matern
import torch
from typing import Tuple
from GP import GP

domain = np.array([[0, 5]])
domain_t = torch.tensor(domain)


""" Solution """


class BO_algo:
    def __init__(self, on_docker=True):
        """Initializes the algorithm with a parameter configuration. """
        self.fast_matrix_inverse = True
        self.on_docker = on_docker

        din = domain.shape[0] # input dimensionality

        # GP for f (accuracy)
        k_f = Matern(length_scale=0.5, nu=2.5)
        self.GP_f = GP(0.0, 0.15, lambda x, y: 0.5 * torch.from_numpy(k_f(x, y)), din)

        # GP for v (speed)
        k_v = Matern(length_scale=0.5, nu=2.5)
        self.GP_v = GP(1.5, 1e-4, lambda x, y: math.sqrt(2.0) * torch.from_numpy(k_v(x, y)), din)

        self.kappa = 1.2 # minimum tolerated speed
        self.beta = 1.0 # used for mu + beta*sigma in acquisition function
        self.gamma = 10.0 # controlling constraint strength in acquisition function
        self.x_init = domain[:, 0] + (domain[:, 1] - domain[:, 0]) / 2.0 # initial point

        self.xs = torch.zeros(0,domain_t.shape[0]).double()
        self.fs = torch.zeros(0).double()
        self.vs = torch.zeros(0).double()

        self.Kf_AA_sig2_inv = torch.zeros(0,0).double()
        self.Kv_AA_sig2_inv = torch.zeros(0,0).double()

    def next_recommendation(self):
        return self.next_recommendation_UCB()

    def next_recommendation_UCB(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        if self.GP_f.num_points == 0:
            x = np.atleast_2d(self.x_init)
        else:
            x = self.optimize_acquisition_function()
        return x


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
                                   approx_grad=True, factr=1e10)  # factr=1e10 => only optimize to 1e10*1e-16=1e-6 accuracy (for speed)
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

        x = torch.tensor(x)
        mu_f, sigma_f = self.GP_f.predict(x)
        mu_v, sigma_v = self.GP_v.predict(x)

        return (mu_f - self.gamma*(self.kappa - mu_v).clamp(min=0.) + self.beta*(sigma_f + sigma_v)).item()


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

        self.GP_f.add_point_and_update_kernel_inverse(x, f)
        self.GP_v.add_point_and_update_kernel_inverse(x, v)


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        xstar = torch.linspace(domain[0, 0], domain[0, 1], steps=1000)
        mu_f, sigma_f = self.GP_f.predict(xstar)
        mu_v, sigma_v = self.GP_v.predict(xstar)
        # docker hack
        if (mu_v.ndim == 2 and mu_v.shape[1] == 1):
            mu_v = mu_v.squeeze(1)
        if (mu_f.ndim == 2 and mu_f.shape[1] == 1):
            mu_f = mu_f.squeeze(1)
        valid_ind = (mu_v - sigma_v) > self.kappa
        x_valid = xstar[valid_ind]
        x_opt = x_valid[mu_f[valid_ind].argmax()].squeeze()
        return x_opt


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
    return (x*np.cos(x)).squeeze()
    #return 2.0

def train_agent(agent, n_iters=20, debug=False):
    # Loop until budget is exhausted
    for j in range(n_iters):
        # Get next recommendation
        x = agent.next_recommendation()
        #x = agent.next_recommendation_thompson()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)
        if debug:
            M1f = agent.Kf_AA_sig2_inv
            xs = agent.xs
            M2f = (agent.Matern_f(xs, xs) + agent.σ_f**2 * torch.eye(xs.shape[0])).inverse()

            M1v = agent.Kv_AA_sig2_inv
            xs = agent.xs
            M2v = (agent.Matern_v(xs, xs) + agent.σ_v**2 * torch.eye(xs.shape[0])).inverse()
            print(f"F/V-error: {(M1f - M2f).abs().sum().item():.2E}, {(M1v - M2v).abs().sum().item():.2E}")


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
            fig, (ax1, ax2) = plt.subplots(1,2)
        xs = torch.linspace(0,5)
        x_best = agent.get_solution()

        # Plot f
        ys = np.array(list(map(f, xs)))
        mus, sigs = agent.get_mu_sigma_f(xs.unsqueeze(1))
        ax1.plot(xs, ys, label="GT")
        ax1.plot(xs, mus, '--', label="Mean")
        ax1.plot(xs, mus+sigs, '-.', c='g', label="Mean + std")
        ax1.plot(xs, mus-sigs, '-.', c='g', label="Mean - std")
        ax1.scatter(agent.xs, agent.fs, label="Sample points")
        ax1.vlines(x_best, ymin=agent.κ-0.3, ymax=agent.κ+0.3, colors='r')
        ax1.legend()
        ax1.set_title("f", fontsize=16)

        # Plot v
        ys = np.array(list(map(v, xs)))
        mus, sigs = agent.get_mu_sigma_v(xs.unsqueeze(1))
        ax2.plot(xs, ys, label="GT")
        ax2.plot(xs, mus, '--', label="Mean")
        ax2.plot(xs, mus+sigs, '-.', c='g', label="Mean + std")
        ax2.plot(xs, mus-sigs, '-.', c='g', label="Mean - std")
        ax2.hlines(agent.κ, xmin=domain[0,0], xmax=domain[0,1])
        ax2.vlines(x_best, ymin=agent.κ-0.3, ymax=agent.κ+0.3, colors='r')
        ax2.scatter(agent.xs, agent.vs, label="Sample points")
        ax2.legend()
        ax2.set_title("v", fontsize=16)
    except ImportError:
        pass

def main():
    # Init problem
    agent = BO_algo(on_docker=False)
    train_agent(agent, debug=True)
    plot_agent(agent)

if __name__ == "__main__":
    main()
