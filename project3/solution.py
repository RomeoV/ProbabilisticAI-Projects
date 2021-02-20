import torch
import math
import numpy as np
from math import sqrt
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process.kernels import Matern
import torch
from typing import Tuple
from GP import GP
from celluloid import Camera
import matplotlib.pyplot as plt

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
        if sum(valid_ind) == 0:
            return torch.tensor([]).double()
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
    return (x*np.cos(x)).squeeze()+1.2
    #return 2.0

def train_agent(agent, n_iters=16, debug=False, create_animation=False):
    if create_animation:
        fig, axes = plt.subplots(1,2)
        camera = Camera(fig)

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
        if create_animation:
            plot_agent(agent, (fig, axes))
            camera.snap()
        if debug:
            M1f = agent.Kf_AA_sig2_inv
            xs = agent.xs
            M2f = (agent.Matern_f(xs, xs) + agent.σ_f**2 * torch.eye(xs.shape[0])).inverse()

            M1v = agent.Kv_AA_sig2_inv
            xs = agent.xs
            M2v = (agent.Matern_v(xs, xs) + agent.σ_v**2 * torch.eye(xs.shape[0])).inverse()
            print(f"F/V-error: {(M1f - M2f).abs().sum().item():.2E}, {(M1v - M2v).abs().sum().item():.2E}")

    if create_animation:
        animation = camera.animate(interval=750)
        animation.save('animation.mp4')


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

def plot_agent(agent, fig_axes=None):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import matplotlib as mpl
        plt.rcParams.update({
            "text.usetex": True,
            # "font.family": "sans-serif",
            # "font.sans-serif": ["Helvetica"]
            })
        if fig_axes == None:
            fig, (ax1, ax2) = fig.subplots(1,2)
            fig = plt.figure()
        else:
            fig, (ax1, ax2) = fig_axes

        xs = torch.linspace(0,5)
        x_best = agent.get_solution()
        if x_best.size() == torch.Size([0]):
            f_best = torch.tensor([])
            v_best = torch.tensor([])
        else:
            f_best, _ = agent.GP_f.predict(x_best.unsqueeze(0))
            v_best, _ = agent.GP_v.predict(x_best.unsqueeze(0))

        x_next = torch.from_numpy(agent.next_recommendation()).squeeze(0)
        f_next, _ = agent.GP_f.predict(x_next)
        v_next, _ = agent.GP_v.predict(x_next)

        # Plot f
        ys = np.array(list(map(f, xs)))
        mus, sigs = agent.GP_f.predict(xs)
        l1, = ax1.plot(xs, ys, c='tab:blue')
        l2, = ax1.plot(xs, mus, '--', c='tab:orange')
        l3, = ax1.plot(xs, mus+sigs, '-.', c='g', linewidth=0.4)
        l4, = ax1.plot(xs, mus-sigs, '-.', c='g', linewidth=0.4)
        ax1.fill_between(xs, mus-sigs, mus+sigs, color='g', alpha=0.3)
        l5 = ax1.scatter(agent.GP_f.xs, agent.GP_f.ys, marker='*', c='black', zorder=100)
        l6 = ax1.vlines(x_next, ymin=f_next-0.3, ymax=f_next+0.3, colors='r')
        # ax1.legend()
        ax1.legend((l1, l2, l3, l4, l5, l6), ('Ground truth', 'Mean', 'Mean + std', 'Mean - std', 'Sample points', 'Next sample'))
        ax1.set_title("f", fontsize=16)
        if x_best.size() > torch.Size([0]):
            ax1.annotate("Best", (x_best, f_best.squeeze()), xytext=(-80, 80), textcoords='offset pixels', arrowprops={'arrowstyle': '->'})

        # Plot v
        ys = np.array(list(map(v, xs)))
        mus, sigs = agent.GP_v.predict(xs)
        l1, = ax2.plot(xs, ys, c='tab:blue')
        l2, = ax2.plot(xs, mus, '--', c='tab:orange')
        l3, = ax2.plot(xs, mus+sigs,  c='g', linewidth=0.4)
        l4, = ax2.plot(xs, mus-sigs,  c='g', linewidth=0.4)
        ax2.fill_between(xs, mus-sigs, mus+sigs, color='g', alpha=0.3)
        ax2.hlines(agent.kappa, xmin=domain[0,0]-0.1, xmax=domain[0,1]+0.1, colors='dimgray')
        ax2.add_patch(
                Rectangle((-0.1, agent.kappa), 5.2, 2, color='g', alpha=0.2, zorder=0.1)
                )
        ax2.add_patch(
                Rectangle((-0.1, agent.kappa-4), 5.2, 4, color='r', alpha=0.2, zorder=0.1)
                )
        # ax2.annotate("valid", (1.7, agent.kappa), xytext=(0,40), textcoords='offset pixels', arrowprops={'arrowstyle': '<-'})
        # ax2.vlines(x_best, ymin=agent.kappa-0.3, ymax=agent.kappa+0.3, colors='r')
        l6 = ax2.vlines(x_next, ymin=v_next-0.3, ymax=v_next+0.3, colors='r')
        if x_best.size() > torch.Size([0]):
            ax2.annotate("Best", (x_best, v_best.squeeze()), xytext=(80, 80), textcoords='offset pixels', arrowprops={'arrowstyle': '->'})
        l5 = ax2.scatter(agent.GP_v.xs, agent.GP_v.ys, marker='*', c='black', zorder=100)
        ax2.legend((l1, l2, l3, l4, l5, l6), ('Ground truth', 'Mean', 'Mean + std', 'Mean - std', 'Sample points', 'Next sample'))
        ax2.set_title("v", fontsize=16)

        plt.gcf().suptitle(r"Find $\mathrm{argmax}_x f(x)$ satisfying $v(x) > \kappa$", fontsize=16)

        plt.tight_layout()
        # plt.show()
    except ImportError:
        pass

def main():
    # Init problem
    agent = BO_algo(on_docker=False)
    train_agent(agent, debug=False, create_animation=True)
    # plot_agent(agent)

if __name__ == "__main__":
    main()
