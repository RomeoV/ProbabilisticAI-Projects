import numpy as np
from GPmodels import ExactGP
from GPmodels import RandomFeatureGP
from gpytorch.kernels.rff_kernel import RFFKernel


import gpytorch as gpt
import torch
# import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)
torch.autograd.set_detect_anomaly(True)

RUN_DOCKER = True
# Constant for Cost function
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
    mask_w1 = np.logical_and(predicted >= true, mask)
    mask_w2 = np.logical_and(np.logical_and(predicted < true, predicted >= THRESHOLD), mask)
    mask_w3 = np.logical_and(predicted < THRESHOLD, mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2
    cost[mask_w3] = cost[mask_w3]*W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted > true, mask)
    mask_w2 = np.logical_and(predicted <= true, mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2

    reward = W4*np.logical_and(predicted < THRESHOLD, true < THRESHOLD)
    if reward is None:
        reward = 0
    return np.mean(cost) - np.mean(reward)


class Model():

    def __init__(self):
        self.learning_rate = 0.1

        self.kernels = "RFF"
        self.composition = None

        self.lengthscale = [0.1]
        self.outputscale = [0.1]
        self._noise = 0.1
        self.num_samples = 400

        self.already_fitted = False

    def predict(self, test_x):
        """
            TODO: enter your code here
        """
        test_x = torch.from_numpy(test_x)
        predictions = self.model(test_x)
        means = predictions.mean.detach()
        std = predictions.variance.detach().sqrt()
        means[means+2*std > 0.5] = 0.5001
        return means.numpy()

    def removeDuplicatesFromList(self, lst):
        return [t for t in (set(tuple(i) for i in lst))]

    def find_duplicate_indices(self, train_x):
        length = train_x.shape[0]
        duplicates = []
        for i in range(length):
            idx = np.where(((train_x[:, 0] == train_x[i][0]) & (train_x[:, 1] == train_x[i][1])))[0]
            duplicates.append(idx)

        unique_duplicates = set(map(tuple, duplicates))
        return unique_duplicates

    def pre_processing(self, train_x, train_y):
        duplicates = self.find_duplicate_indices(train_x)
        train_y_new = []
        train_x_new = []

        for tuple_ in duplicates:
            mean = train_y[[tuple_]].mean()
            train_y_new.append(mean)
            train_x_new.append(train_x[tuple_[0]])

        # np.delete(first_value, train_x[])
        return (np.array(train_x_new), np.array(train_y_new))

    def fit_model(self, train_x, train_y):
        """
             TODO: enter your code here
        Computes Kernel inverse for training data

        See PAI slideset 3, page 32
        $\mu'(x) = \mu(x) + k_{x,A} (K_{AA} + \sigma^2 I)^{-1} (y_A - \mu_A)$

        First, we compute our data prior.
        Then, we store our training inputs (xs) to compute $k_{x,A}$, later.
        Then, we compute the (approximate) inverse to the Kernel matrix + prior.
        The problem is that the kernel matrix might be too large to fit into memory (or even invert).
        For this, we potentially employ some approximation, like the nystrom method.
        The inverse times the label data is stored in `self.alpha`.
        """
        train_x, train_y = self.pre_processing(train_x, train_y)
        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y)

        kernel = self.get_kernel(self.kernels, self.composition)
        kernel = gpt.kernels.ScaleKernel(RFFKernel(4000))
        likelihood = gpt.likelihoods.GaussianLikelihood()

        # self.model = RandomFeatureGP(train_x, train_y, kernel, likelihood)
        self.model = ExactGP(train_x, train_y, kernel, likelihood)

        self.model.train()
        # self.model = RandomFeatureGP(train_x,
        #                              train_y,
        #                              kernel,
        #                              self.num_samples,
        #                              "RFF")
        # ===================================================
        #        Setting the hyperparameters
        # ===================================================
        self.model.lengthscale = self.lengthscale
        self.model.outputscale = self.outputscale
        self.model.likelihood.noise = self.noise
        # 'covar_module.base_kernel.kernels.0': torch.tensor(0.5),  #
        hypers = {
            'likelihood.noise_covar.noise': torch.tensor(0.044),
            'mean_module.constant': torch.tensor(0.35),
            'covar_module.base_kernel.lengthscale': torch.tensor(0.22),
            'covar_module.outputscale': torch.tensor(0.033),
        }
        self.model.initialize(**hypers)
        # ===================================================
        #  Fitting the Hyperparameters Calcualting the model
        # ===================================================
        self.model_selection(train_x, train_y)
        self.model.eval()

        self.already_fitted = True
        if False:
            self.plot_model(train_x, train_y)

    def model_selection(self,
                        train_x,
                        train_y,
                        optimizer=torch.optim.Adam,
                        learning_rate=0.1,
                        training_iter=20):
        optimizer = optimizer([{'params': self.model.parameters()}],
                              lr=learning_rate)
        mll = gpt.mlls.ExactMarginalLogLikelihood(self.model.likelihood,
                                                  self.model)

        losses = []
        lengthscales = []
        outputscales = []
        noises = []

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward(retain_graph=True)

            losses.append(loss.item())
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f  mean: %.3f' % (
                i + 1, training_iter, loss.item(),
                self.model.covar_module.base_kernel.lengthscale.item(),
                self.model.likelihood.noise.item(),
                self.model.mean_module.constant.item()
            ))
            lengthscales.append(self.model.lengthscale.item())
            outputscales.append(self.model.outputscale.item())
            noises.append(self.model.likelihood.noise.item())
            optimizer.step()

        print("Finished Training")

    def get_kernel(self, kernel, composition="addition", **kwargs):
        base_kernel = []
        if "RBF" in kernel:
            base_kernel.append(gpt.kernels.RBFKernel())
        if "linear" in kernel:
            base_kernel.append(gpt.kernels.LinearKernel())
        if "quadratic" in kernel:
            base_kernel.append(gpt.kernels.PolynomialKernel(power=2))
        if "Matern-1/2" in kernel:
            base_kernel.append(gpt.kernels.MaternKernel(nu=1/2))
        if "Matern-3/2" in kernel:
            base_kernel.append(gpt.kernels.MaternKernel(nu=3/2))
        if "Matern-5/2" in kernel:
            base_kernel.append(gpt.kernels.MaternKernel(nu=5/2))
        if "Cosine" in kernel:
            base_kernel.append(gpt.kernels.CosineKernel())
        if "RFF":
            base_kernel.append(gpt.kernels.ScaleKernel(RFFKernel(num_samples=kwargs.get("num_samples", 8000))))

        if composition == "addition":
            base_kernel = gpt.kernels.AdditiveKernel(*base_kernel)
        elif composition == "product":
            base_kernel = gpt.kernels.ProductKernel(*base_kernel)
        elif composition is None:
            base_kernel = base_kernel[0]
        else:
            raise NotImplementedError
        kernel = gpt.kernels.ScaleKernel(base_kernel)
        return kernel

    @property
    def noise(self):
        """Get liklihood nosie"""
        return self._noise

    @noise.setter
    def noise(self, value):
        """Set noise level"""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value])
        self._noise = value

    def plot_model(self, train_x, train_y, inducing_points=None, plot_points=True):
        assert self.already_fitted, "Model has to be fitted first!"

        with torch.no_grad():
            out = self.model(train_x)
            pred = out.mean.numpy()
            lower, upper = out.confidence_region()

        # if plot_points:
        #     plt.plot(train_x, train_y, 'k*', label='Train Data')
        # plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, marker='o')
        plt.contourf(train_x[:, 0], train_x[:, 1],  pred, levels=16)
        plt.colorbar()
        # plt.pyplot.contourf(train_x, train_x,  pred, levels=16)
        plt.show()


def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    if not RUN_DOCKER:
        train_x = np.loadtxt(train_x_name, delimiter=',')
        train_y = np.loadtxt(train_y_name, delimiter=',')

        # # load the test dateset
        test_x_name = "test_x.csv"
        test_x = np.loadtxt(test_x_name, delimiter=',')
    else:
        train_x = torch.tensor(np.loadtxt(train_x_name, delimiter=','))
        train_y = torch.tensor(np.loadtxt(train_y_name, delimiter=','))

        test_x_name = "test_x.csv"
        test_x = torch.tensor(np.loadtxt(test_x_name, delimiter=','))

    M = Model()
    M.fit_model(train_x, train_y)

    prediction = M.predict(test_x)

    print(prediction)


if __name__ == "__main__":
    main()
