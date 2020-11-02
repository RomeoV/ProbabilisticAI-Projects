import numpy as np
import torch
import os
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
import torch
from torch.nn import functional as F
from tqdm import trange, tqdm


def ece(probs, labels, n_bins=30):
    '''
    probs has shape [n_examples, n_classes], labels has shape [n_class] -> np.float
    Computes the Expected Calibration Error (ECE). Many options are possible,
    in this implementation, we provide a simple version.

    Using a uniform binning scheme on the full range of probabilities, zero
    to one, we bin the probabilities of the predicted label only (ignoring
    all other probabilities). For the ith bin, we compute the avg predicted
    probability, p_i, and the bin's total accuracy, a_i. We then compute the
    ith calibration error of the bin, |p_i - a_i|. The final returned value
    is the weighted average of calibration errors of each bin.
    '''
    n_examples, n_classes = probs.shape

    # assume that the prediction is the class with the highest prob.
    preds = np.argmax(probs, axis=1)

    onehot_labels = np.eye(n_classes)[labels]

    predicted_class_probs = probs[range(n_examples), preds]

    # Use uniform bins on the range of probabilities, i.e. closed interval [0.,1.]
    bin_upper_edges = np.histogram_bin_edges([], bins=n_bins, range=(0., 1.))
    bin_upper_edges = bin_upper_edges[1:] # bin_upper_edges[0] = 0.

    probs_as_bin_num = np.digitize(predicted_class_probs, bin_upper_edges)
    sums_per_bin = np.bincount(probs_as_bin_num, minlength=n_bins, weights=predicted_class_probs)
    sums_per_bin = sums_per_bin.astype(np.float32)

    total_per_bin = np.bincount(probs_as_bin_num, minlength=n_bins) \
        + np.finfo(sums_per_bin.dtype).eps # division by zero
    avg_prob_per_bin = sums_per_bin / total_per_bin

    accuracies = onehot_labels[range(n_examples), preds] # accuracies[i] is 0 or 1
    accuracies_per_bin = np.bincount(probs_as_bin_num, weights=accuracies, minlength=n_bins) \
        / total_per_bin

    prob_of_being_in_a_bin = total_per_bin / float(n_examples)

    ece_ret = np.abs(accuracies_per_bin - avg_prob_per_bin) * prob_of_being_in_a_bin
    ece_ret = np.sum(ece_ret)
    return ece_ret


def load_rotated_mnist():
    '''
    The difference between MNIST and Rotated MNIST is that Rotated MNIST has
    rotated *test* images.
    '''

    mnist_path = "/data/rotated_mnist.npz"
    if not os.path.isfile(mnist_path):
        mnist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/rotated_mnist.npz")

    data = np.load(mnist_path)

    x_train = torch.from_numpy(data["x_train"]).reshape([-1, 784])
    y_train = torch.from_numpy(data["y_train"])

    dataset_train = torch.utils.data.TensorDataset(x_train, y_train)

    return dataset_train


class Densenet(torch.nn.Module):
    '''
    Simple module implementing a feedforward neural network with
    num_layers layers of size width and input of size input_size.
    '''
    def __init__(self, input_size, num_layers, width):
        super().__init__()
        input_layer = torch.nn.Sequential(nn.Linear(input_size, width),
                                           nn.ReLU())
        hidden_layers = [nn.Sequential(nn.Linear(width, width),
                                    nn.ReLU()) for _ in range(num_layers)]
        output_layer = torch.nn.Linear(width, 10)
        layers = [input_layer, *hidden_layers, output_layer]
        self.net = torch.nn.Sequential(*layers)


    def forward(self, x):
        out = self.net(x)
        return out


    def predict_class_probs(self, x):
        probs = F.softmax(self.forward(x), dim=1)
        return probs


class BayesianLayer(torch.nn.Module):
    '''
    Module implementing a single Bayesian feedforward layer.
    The module performs Bayes-by-backprop, that is, mean-field
    variational inference. It keeps prior and posterior weights
    (and biases) and uses the reparameterization trick for sampling.
    '''

    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = bias

        # TODO: enter your code here
        self.prior_mu = torch.tensor([0.], requires_grad=False)  # this we just guess
        #self.prior_sigma = torch.tensor([0.1], requires_grad=False)
        self.prior_sigma = torch.tensor([1.], requires_grad=False).sqrt()
        self.weight_mu = nn.Parameter(torch.zeros(output_dim, input_dim))
        self.weight_logsigma = nn.Parameter(torch.zeros(output_dim, input_dim))
        nn.init.normal_(self.weight_mu, self.prior_mu.item(), self.prior_sigma.pow(2).item())
        nn.init.normal_(self.weight_logsigma, self.prior_mu.item(), (self.prior_sigma.pow(2).item()/10))

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.zeros(output_dim))
            self.bias_logsigma = nn.Parameter(torch.ones(output_dim))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_logsigma', None)


    def forward(self, inputs):
        # inputs will be (n_batch x n_features)
        # 1) Sample weights
        # 2) Sample bias
        # 3) Apply linear layer using these weights
        weight_distribution = torch.distributions.Normal(
                self.weight_mu,
                self.weight_logsigma.exp().pow(2),
        )
        if self.use_bias:
            bias_distribution = torch.distributions.Normal(
                    self.bias_mu,
                    self.bias_logsigma.exp().pow(2),
            )

        normal = torch.distributions.Normal(0, 0.05)

        if self.use_bias:
            # TODO: enter your code here
            eps_W = normal.sample(sample_shape=self.weight_mu.size())
            eps_b = normal.sample(sample_shape=(self.weight_mu.size()[0],))
            #eps_W = eps_W.mean(0)
            #eps_b = eps_b.mean(0)
            W = self.weight_logsigma.exp() * eps_W + self.weight_mu
            b = self.bias_logsigma.exp() * eps_b + self.bias_mu
            #W = weight_distribution.rsample()
            #b = bias_distribution.rsample()
            y = inputs @ W.t() + b
        else:
            bias = None
            eps_W = normal.sample(sample_shape=self.weight_mu.size()).requires_grad_(False)
            W = self.weight_logsigma.exp() * eps_W + self.weight_mu
            #W = weight_distribution.rsample()
            y = inputs @ W.t()

        return y


    def kl_divergence(self):
        '''
        Computes the KL divergence between the priors and posteriors for this layer.
        '''
        kl_loss = self._kl_divergence(self.weight_mu, self.weight_logsigma)
        if self.use_bias:
            kl_loss += self._kl_divergence(self.bias_mu, self.bias_logsigma)
        return kl_loss


    def _kl_divergence(self, mu, logsigma):
        '''
        Computes the KL divergence between one Gaussian posterior
        and the Gaussian prior.
        '''

        d = mu.numel()
        # TODO: enter your code here

        """ Full gaussian kl-divergence """
        kl_gauss_full = 1/2 * (1/self.prior_sigma.pow(2) * logsigma.exp().pow(2).sum()
                    + (self.prior_mu - mu).pow(2).sum() / (self.prior_sigma.pow(2))
                    - d
                    + (d*torch.log(self.prior_sigma**2) - 2*logsigma.sum())
                    )

        """ Reduced gaussian kl-divergence """
        kl_gauss_simpl = 1/2 * (logsigma.exp().pow(2).sum() + mu.pow(2).sum() - d - 2*logsigma.sum())

        #assert (kl_gauss_full - kl_gauss_simpl).abs() < 1e-4, f"{kl_gauss_full.item():.4f}, {kl_gauss_simpl.item():.4f}"

        # print((logsigma.abs().min().item(), logsigma.abs().max().item()))

        return kl_gauss_full


class BayesNet(torch.nn.Module):
    '''
    Module implementing a Bayesian feedforward neural network using
    BayesianLayer objects.
    '''

    def __init__(self, input_size, num_layers, width, temp=1):
        super().__init__()
        input_layer = torch.nn.Sequential(BayesianLayer(input_size, width),
                                           nn.ReLU())
        hidden_layers = [nn.Sequential(BayesianLayer(width, width),
                                    nn.ReLU()) for _ in range(num_layers)]
        output_layer = BayesianLayer(width, 10)
        layers = [input_layer, *hidden_layers, output_layer]
        self.net = torch.nn.Sequential(*layers)
        self.temp = temp
        print(self.net)


    def forward(self, x):
        return self.net(x)


    def predict_class_probs(self, x, num_forward_passes=10):
        assert x.shape[1] == 28**2
        batch_size = x.shape[0]

        # TODO: make n random forward passes
        # compute the categorical softmax probabilities
        # marginalize the probabilities over the n forward passes

        results = torch.zeros(num_forward_passes, batch_size, 10)
        for i in range(num_forward_passes):
            results[i] = self.forward(x)
        probs = (results.mean(0)/self.temp).softmax(-1)

        assert probs.shape == (batch_size, 10)
        assert (probs[0,:].sum() - 1).abs() < 1e-5, f"{probs[0,:].sum():.3f}"
        return probs


    def kl_loss(self):
        '''
        Computes the KL divergence loss for all layers.
        '''
        # TODO: enter your code here
        kl_divergences_prior = (sum(
            l[0].kl_divergence() for l in self.net[:-1])
            + self.net[-1].kl_divergence())

        return kl_divergences_prior

def train_network(model, optimizer, scheduler, train_loader, num_epochs=100, pbar_update_interval=100, val_loader=None):
    '''
    Updates the model parameters (in place) using the given optimizer object.
    Returns `None`.

    The progress bar computes the accuracy every `pbar_update_interval`
    iterations.
    '''

    criterion = torch.nn.CrossEntropyLoss()  # always used in this assignment

    pbar = trange(num_epochs)
    M = max(k for (k, (batch_x, batch_y)) in enumerate(train_loader))
    for i in pbar:
        for k, ((batch_x, batch_y), (batch_x_val, batch_y_val)) in enumerate(zip(train_loader, val_loader)):
            model.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            if type(model) == BayesNet:
                # BayesNet implies additional KL-loss.
                # TODO: enter your code here
                pi = (2**(M-(k+1)))/(2**M - 1)
                loss += pi * model.kl_loss().squeeze()
                if i%5 == 0 and k == 0:
                    print(f"Loss (likelihood, complexity, pi): {criterion(y_pred, batch_y):.3f}, {model.kl_loss().squeeze():.3f}, {pi:.3f}")
                    if val_loader:
                        evaluate_model(model, 'bayesnet', val_loader, 2000, False, False)
                    torch.save(model, f"models/model-{i}")
            loss.backward()
            optimizer.step()

            if k % pbar_update_interval == 0:
                acc = (model.predict_class_probs(batch_x).argmax(axis=1) == batch_y).sum().float()/(len(batch_y))
                acc_val = (model.predict_class_probs(batch_x_val).argmax(axis=1) == batch_y_val).sum().float()/(len(batch_y_val))
                pbar.set_postfix(loss=loss.item(), acc=acc.item(), val_acc=acc_val.item())

        scheduler.step()


def evaluate_model(model, model_type, test_loader, batch_size, extended_eval, private_test):
    '''
    Evaluates the trained model based on accuracy and ECE.
    If extended_eval is True, also computes predictive confidences
    on the FashionMNIST data set (out-of-distribution/OOD) and saves the
    most and least confidently classified images for both data sets
    as well as the classification performance for OOD detection based
    on the predictive confidences.
    '''
    accs_test = []
    probs = torch.tensor([])
    labels = torch.tensor([]).long()
    for batch_x, batch_y in test_loader:
        pred = model.predict_class_probs(batch_x)
        probs = torch.cat((probs, pred))
        labels = torch.cat((labels, batch_y))
        acc = (pred.argmax(axis=1) == batch_y).sum().float().item()/(len(batch_y))
        accs_test.append(acc)

    if not private_test:
        acc_mean = np.mean(accs_test)
        ece_mean = ece(probs.detach().numpy(), labels.numpy())
        print(f"Model type: {model_type}\nAccuracy = {acc_mean:.3f}\nECE = {ece_mean:.3f}")
    else:
        print("Using private test set.")

    final_probs = probs.detach().numpy()

    if extended_eval:
        confidences = []
        for batch_x, batch_y in test_loader:
            pred = model.predict_class_probs(batch_x)
            confs, _ = pred.max(dim=1)
            confidences.extend(confs.detach().numpy())

        confidences = np.array(confidences)

        fig, axs = plt.subplots(ncols=10, figsize=(20,2))
        for ax, idx in zip(axs, confidences.argsort()[-10:]):
            ax.imshow(test_loader.dataset.tensors[0][idx].numpy().reshape((28,28)), cmap="gray")
            ax.axis("off")
        fig.suptitle("Most confident predictions", size=20);
        fig.savefig(f"mnist_most_confident_{model_type}.pdf")

        fig, axs = plt.subplots(ncols=10, figsize=(20,2))
        for ax, idx in zip(axs, confidences.argsort()[:10]):
            ax.imshow(test_loader.dataset.tensors[0][idx].numpy().reshape((28,28)), cmap="gray")
            ax.axis("off")
        fig.suptitle("Least confident predictions", size=20);
        fig.savefig(f"mnist_least_confident_{model_type}.pdf")

        fmnist_path = "/data/fashion/fmnist.npz"
        if not os.path.isfile(fmnist_path):
            fmnist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/fashion/fmnist.npz")
        data_fmnist = np.load(fmnist_path)["x_test"]
        dataset_fmnist = torch.utils.data.TensorDataset(torch.tensor(data_fmnist))
        dataloader_fmnist = torch.utils.data.DataLoader(dataset_fmnist, batch_size=batch_size)

        confidences_fmnist = []
        for batch_x in dataloader_fmnist:
            pred = model.predict_class_probs(batch_x[0])
            confs, _ = pred.max(dim=1)
            confidences_fmnist.extend(confs.detach().numpy())

        confidences_fmnist = np.array(confidences_fmnist)

        fig, axs = plt.subplots(ncols=10, figsize=(20,2))
        for ax, idx in zip(axs, confidences_fmnist.argsort()[-10:]):
            ax.imshow(dataloader_fmnist.dataset.tensors[0][idx].numpy().reshape((28,28)), cmap="gray")
            ax.axis("off")
        fig.suptitle("Most confident predictions", size=20);
        fig.savefig(f"fashionmnist_most_confident_{model_type}.pdf")

        fig, axs = plt.subplots(ncols=10, figsize=(20,2))
        for ax, idx in zip(axs, confidences_fmnist.argsort()[:10]):
            ax.imshow(dataloader_fmnist.dataset.tensors[0][idx].numpy().reshape((28,28)), cmap="gray")
            ax.axis("off")
        fig.suptitle("Least confident predictions", size=20);
        fig.savefig(f"fashionmnist_least_confident_{model_type}.pdf")

        confidences_all = np.concatenate([confidences, confidences_fmnist])
        dataset_labels = np.concatenate([np.ones_like(confidences), np.zeros_like(confidences_fmnist)])

        print(f"AUROC for MNIST vs. FashionMNIST OOD detection based on {model_type} confidence: "
              f"{roc_auc_score(dataset_labels, confidences_all):.3f}")
        print(f"AUPRC for MNIST vs. FashionMNIST OOD detection based on {model_type} confidence: "
              f"{average_precision_score(dataset_labels, confidences_all):.3f}")

    return final_probs


def main(test_loader=None, private_test=False):
    num_epochs = 100 # You might want to adjust this
    batch_size = 2000  # Try playing around with this
    print_interval = 200
    learning_rate = 1e-3  # Try playing around with this
    model_type = "bayesnet"  # Try changing this to "densenet" as a comparison
    extended_evaluation = False  # Set this to True for additional model evaluation

    dataset_train = load_rotated_mnist()
    dataset_train, dataset_val = torch.utils.data.random_split(dataset_train, [55000, 5000])
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                               shuffle=True, drop_last=True)
    val_loader   = torch.utils.data.DataLoader(dataset_val  , batch_size=batch_size//20,
                                               shuffle=True, drop_last=True)
    torch.save(val_loader, 'val_loader')

    if model_type == "bayesnet":
        model = BayesNet(input_size=784, num_layers=2, width=100)
    elif model_type == "densenet":
        model = Densenet(input_size=784, num_layers=2, width=100)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=(30,60), gamma=0.5)
    train_network(model, optimizer, scheduler, train_loader,
                 num_epochs=num_epochs, pbar_update_interval=print_interval, val_loader=val_loader)
    torch.save(model, 'model')

    if test_loader is None:
        print("evaluating on train data")
        test_loader = train_loader
    else:
        print("evaluating on test data")

    # Do not change this! The main() method should return the predictions for the test loader
    predictions = evaluate_model(model, model_type, test_loader, batch_size, extended_evaluation, private_test)
    return predictions


if __name__=="__main__":
    main()
