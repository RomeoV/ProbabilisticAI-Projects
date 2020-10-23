import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from NystromSolver import NystromSolver

## Constant for Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted)**2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted>=true,mask)
    mask_w2 = np.logical_and(np.logical_and(predicted<true,predicted >=THRESHOLD),mask)
    mask_w3 = np.logical_and(predicted<THRESHOLD,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2
    cost[mask_w3] = cost[mask_w3]*W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted>true,mask)
    mask_w2 = np.logical_and(predicted<=true,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2

    reward = W4*np.logical_and(predicted < THRESHOLD,true<THRESHOLD)
    if reward is None:
        reward = 0
    return np.mean(cost) - np.mean(reward)

"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""


class Model():

    def __init__(self):
        """
            Sets kernel type and hyperparameters
        """
        # Setup approximation parameters
        self.kernel = gpytorch.kernels.RBFKernel()
        self.m = 300

    def predict(self, test_x):
        """
            Makes predictions.
        """
        # Load test_x into a torch tensor
        test_x = torch.from_numpy(test_x)

        # Compute k_xA
        k_xA = self.kernel(test_x, self.train_x_perm).evaluate()

        # Compute mean
        # both train_x and alpha are permuted so multiplication should be fine
        mu = self.mu_prior + k_xA @ self.alpha

        # Compute covariance (solve with multiple right-hand-sides)
        beta = self.solver.solve(k_xA.t()) # rhs already permuted
        # beta = beta[self.iperm, :]
        cov = self.kernel(test_x).evaluate().diag() - (k_xA @ beta).diag()

        # Postprocess prediction
        mu[(mu < 0.5).logical_and(mu + 2.5 * torch.sqrt(cov) > 0.5)] = 0.50001

        return mu.detach().numpy()

    def fit_model(self, train_x, train_y):
        """
            Fits GP model.
        """
        # Load train_x and train_y into torch tensors
        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y).unsqueeze(1)
        n = train_x.shape[0]

        # Compute mu and sigma priors based on train_y
        self.mu_prior = train_y.mean()
        self.sigma_prior = train_y.std()

        # Setup permutation vector
        # perm = torch.randperm(n)
        # Use all data from sparse region in NystromSolver
        is_in_sparse_region = train_x[:, 0] > -0.5
        is_in_dense_region = is_in_sparse_region.logical_not()
        index = torch.arange(n)
        index_sparse_region = index[is_in_sparse_region]
        index_dense_region = index[is_in_dense_region]
        # Shuffle data from dense region
        index_dense_region = index_dense_region[torch.randperm(is_in_dense_region.sum())]
        perm = torch.cat((index_sparse_region, index_dense_region), 0)

        # Setup inverse permutation vector
        # iperm = torch.zeros(n, dtype=perm.dtype)
        # iperm[perm] = torch.arange(n)

        # Shuffle data for NystromSolver
        train_x_perm = train_x[perm, :]
        train_y_perm = train_y[perm, :]

        # Setup NystromSolver
        solver = NystromSolver(train_x_perm, self.kernel, self.m, self.sigma_prior**2)
        solver.preprocess()

        # Fit model using Nystrom
        alpha = solver.solve(train_y_perm - self.mu_prior)
        # alpha = alpha[iperm, :].squeeze()
        alpha = alpha.squeeze() # Keep permuted

        # Store data in class
        self.train_x_perm = train_x_perm
        # self.iperm = iperm
        self.solver = solver
        self.alpha = alpha


def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

    print(prediction)

    # Plot predictions
    plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, marker='o')
    plt.scatter(test_x[:, 0], test_x[:, 1], c=prediction, marker='x')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
