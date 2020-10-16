import numpy as np
import torch
import gpytorch as gpt
import matplotlib.pyplot as plt
from gpytorch.kernels.rff_kernel import RFFKernel
from sklearn.model_selection import KFold, train_test_split

torch.set_default_tensor_type(torch.DoubleTensor)

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

class Model:
    def __init__(self):
        self.BHM: BlazingHotModel

    def fit_model(self, train_x, train_y):
        N = train_x.shape[0]
        rand_ind = torch.randperm(N)
        train_ind = rand_ind[:N*8//10]
        val_ind = rand_ind[N*8//10:]
        self.BHM = BlazingHotModel(torch.tensor(train_x), torch.tensor(train_y), train_ind, val_ind)

    def predict(self, test_x):
        predictions = self.BHM(torch.tensor(test_x))
        means = predictions.mean.detach()
        std = predictions.variance.detach().sqrt()
        means[means+2*std > 0.5] = 0.5001
        return means.numpy()


class BlazingHotModel(gpt.models.ExactGP):

    def __init__(self, train_x, train_y, train_ind, val_ind):
        """
            TODO: enter your code here
        """
        likelihood = gpt.likelihoods.GaussianLikelihood()
        super(BlazingHotModel, self).__init__(train_x[train_ind], train_y[train_ind], likelihood)

        self.train_ind = train_ind
        self.val_ind = val_ind

        self.likelihood = likelihood
        self.mean_module = gpt.means.ConstantMean()
        #self.covar_module = gpt.kernels.ScaleKernel(gpt.kernels.RBFKernel())
        self.covar_module = gpt.kernels.ScaleKernel(RFFKernel(num_samples=5000))

        hypers = {
            'likelihood.noise_covar.noise': torch.tensor(0.05),
            'mean_module.constant': torch.tensor(0.6),
            'covar_module.base_kernel.lengthscale': torch.tensor(0.4),
            'covar_module.outputscale': torch.tensor(0.4),
        }

        self.initialize(**hypers)

        self.fit_model(train_x, train_y, training_iter=2)
        self.eval()
        self.likelihood.eval()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpt.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, test_x):
        """
            TODO: enter your code here
        """
        ## dummy code below
        self.eval()
        self.likelihood.eval()
        predictions = self.forward(test_x)
        means = predicions.mean.detach()
        std = predictions.variance.detach().sqrt()
        means[means+2*std > 0.5] = 0.5001
        return means

    def fit_model(self, train_x, train_y, training_iter=20):
        """
             TODO: enter your code here
        """
        self.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.parameters()}
        ], lr=0.01, weight_decay=1e-3)

        mll = gpt.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(training_iter):
            train_x_, val_x_ = train_x[self.train_ind], train_x[self.val_ind]
            train_y_, val_y_ = train_y[self.train_ind], train_y[self.val_ind]

            optimizer.zero_grad()
            output = self.forward(train_x_)

            loss = -mll(output, train_y_)
            loss.backward()

            val_output = self.forward(val_x_)
            val_loss = -mll(val_output, val_y_)
            print(('Iter %d/%d - Loss: %.3f   Val Loss: %.3f   lengthscale: %.3f   outputscale: %.3f   noise: %.3f   mean: %.3f' % (
                i + 1, training_iter,
                loss.item(),
                val_loss.item(),
                self.covar_module.base_kernel.lengthscale.item(),
                self.covar_module.outputscale.item(),
                self.likelihood.noise.item(),
                self.mean_module.constant.item(),
                )
                #+ "   ".join([f"{param} = {val}" for param, val in self.named_parameters()])
            ))

            optimizer.step()


def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    # Convert to torch
    train_x = torch.tensor(train_x).double()
    train_y = torch.tensor(train_y).double()
    test_x = torch.tensor(test_x).double()

    N = 10000
    ind = torch.randperm(train_x.shape[0])
    train_x = train_x[ind[:N],:]
    train_y = train_y[ind[:N]]
    test_x = test_x

    M = Model()
    M.fit_model(train_x, train_y)

    predictions = M.predict(test_x)

    #print(predictions)

    plot_predictions(train_x, train_y, test_x, predictions)

def plot_predictions(train_x, train_y, test_x, pred_y):
    plt.scatter(train_x[:,0], train_x[:,1], c=train_y, marker='o')
    plt.scatter(test_x[:,0], test_x[:,1], c=pred_y, marker='x')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
