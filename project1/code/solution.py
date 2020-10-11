import numpy as np
import torch
import gpytorch as gpt
import matplotlib.pyplot as plt

THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.01


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted) ** 2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted > true, mask)
    mask_w2 = np.logical_and(np.logical_and(predicted < true, predicted > THRESHOLD), mask)
    mask_w3 = np.logical_and(predicted < THRESHOLD, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2
    cost[mask_w3] = cost[mask_w3] * W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted > true, mask)
    mask_w2 = np.logical_and(predicted < true, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2

    # reward for correctly identified safe regions
    reward = W4 * np.logical_and(predicted <= THRESHOLD, true <= THRESHOLD)

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
            TODO: enter your code here
        """

        self.PRIOR_CONST: float  # is computed in fit later
        self.SIGMA = 0.05  # hyperparam
        self.kernel = gpt.kernels.RBFKernel();  # later, put hyperparams here

        pass

    def prior_func(self, x: torch.tensor) -> torch.tensor:
        return self.PRIOR_CONST * torch.ones(x.shape[0])

    def predict(self, test_x):
        """
            TODO: enter your code here
        """
        k_xA = self.kernel(test_x, self.train_x).evaluate()
        mu = self.prior_func(test_x) + (k_xA @ self.alpha).squeeze(1)
        return mu

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        """
        N = train_x.shape[0]
        self.train_x = train_x
        self.PRIOR_CONST = train_y.median();
        mu_A = self.prior_func(train_x)
        K_plus_sigma = self.kernel(train_x, train_x).evaluate() + self.SIGMA**2 * torch.eye(N)
        self.alpha, _ = torch.solve((train_y - mu_A).unsqueeze(1), K_plus_sigma)  # replace with e.g. nystrom
        #import IPython; IPython.embed(); IPython.exit()


def plot(train_x, train_y, test_x, pred_y):
    pred_y = pred_y.detach().numpy()

    plt.scatter(train_x[:,0], train_x[:,1], c=train_y, marker='o')
    plt.scatter(test_x[:,0], test_x[:,1], c=pred_y, marker='x')
    plt.colorbar()
    plt.show()

def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = torch.tensor(np.loadtxt(train_x_name, delimiter=','))
    train_y = torch.tensor(np.loadtxt(train_y_name, delimiter=','))

    ind = torch.randperm(train_x.shape[0])
    train_x = train_x[ind]
    train_y = train_y[ind]

    train_x = train_x[:1000]
    train_y = train_y[:1000]
    #import IPython; IPython.embed(); IPython.exit()

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = torch.tensor(np.loadtxt(test_x_name, delimiter=','))

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

    print(prediction)
    plot(train_x, train_y, test_x, prediction)


if __name__ == "__main__":
    main()
