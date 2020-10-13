import gpytorch
from gpytorch.lazy import MatmulLazyTensor, lazify
from gpytorch.models.exact_prediction_strategies import (
    DefaultPredictionStrategy,
)
from scipy.stats.distributions import chi
import torch
import numpy as np
torch.set_default_tensor_type(torch.DoubleTensor)


class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        super().__init__(train_x, train_y, likelihood=likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        """Forward computation of GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def output_scale(self):
        """Get output scale."""
        return self.covar_module.outputscale

    @output_scale.setter
    def output_scale(self, value):
        """Set output scale."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value])
        self.covar_module.outputscale = value

    @property
    def length_scale(self):
        """Get length scale."""
        ls = self.covar_module.base_kernel.kernels[0].lengthscale
        if ls is None:
            ls = torch.tensor(0.0)
        return ls

    @length_scale.setter
    def length_scale(self, value):
        """Set length scale."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value])

        try:
            self.covar_module.lengthscale = value
        except RuntimeError:
            pass

        try:
            self.covar_module.base_kernel.lengthscale = value
        except RuntimeError:
            pass

        try:
            for kernel in self.covar_module.base_kernel.kernels:
                kernel.lengthscale = value
        except RuntimeError:
            pass


class RandomFeatureGP(ExactGP):
    """
        RFF: Random Fourier Features
        QFF: Quadrature Fourier Features
    """

    def __init__(
        self,
        train_x,
        train_y,
        kernel,
        num_samples,
        approximation="RFF",
    ):
        super().__init__(train_x, train_y, kernel)
        self.num_samples = num_samples
        self.approximation = approximation

        self.dim = train_x.shape[-1]
        self.w, self.b, self._feature_scale = self._sample_features()
        self.full_predictive_covariance = True

    @property
    def scale(self):
        """Return feature scale."""
        return torch.sqrt(self._feature_scale * self.output_scale)

    def sample_features(self):
        """Sample a new set of features."""
        self.w, self.b, self._feature_scale = self._sample_features()

    def _sample_features(self):
        """Sample a new set of random features."""
        # Only squared-exponential kernels are implemented.
        if self.approximation == "RFF":
            w = torch.randn(self.num_samples, self.dim) / torch.sqrt(self.length_scale)
            scale = torch.tensor(1.0 / self.num_samples)

        elif self.approximation == "OFF":
            q, _ = torch.qr(torch.randn(self.num_samples, self.dim))
            diag = torch.diag(
                torch.tensor(
                    chi.rvs(df=self.num_samples, size=self.num_samples),
                    dtype=torch.get_default_dtype(),
                )
            )
            w = (diag @ q) / torch.sqrt(self.length_scale)
            scale = torch.tensor(1.0 / self.num_samples)

        elif self.approximation == "QFF":
            q = int(np.floor(np.power(self.num_samples, 1.0 / self.dim)))
            self._num_samples = q ** self.dim
            omegas, weights = np.polynomial.hermite.hermgauss(2 * q)
            omegas = torch.tensor(omegas[:q], dtype=torch.get_default_dtype())
            weights = torch.tensor(
                weights[:q], dtype=torch.get_default_dtype())

            omegas = torch.sqrt(1.0 / self.length_scale) * omegas
            w = torch.cartesian_prod(*[omegas.squeeze()
                                       for _ in range(self.dim)])
            if self.dim == 1:
                w = w.unsqueeze(-1)

            weights = 4 * weights / np.sqrt(np.pi)
            scale = torch.cartesian_prod(*[weights for _ in range(self.dim)])
            if self.dim > 1:
                scale = scale.prod(dim=1)
        else:
            raise NotImplementedError(f"{self.approximation} not implemented.")

        b = 2 * torch.tensor(np.pi) * torch.rand(self.num_samples)
        self.prediction_strategy = None  # reset prediction strategy.
        return w, b, scale

    def __call__(self, x):
        """Return GP posterior at location `x'."""
        train_inputs = torch.zeros(2 * self.num_samples, 1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        inputs = x

        if self.prediction_strategy is None:
            x = self.train_inputs[0]
            zt = self.forward(x).transpose(-2, -1)

            mean = train_inputs.squeeze(-1)

            cov = lazify(zt @ zt.transpose(-1, -2)).add_jitter()

            y = self.train_targets - self.mean_module(x)
            labels = zt @ y

            prior_dist = gpytorch.distributions.MultivariateNormal(mean, cov)
            self.prediction_strategy = DefaultPredictionStrategy(
                train_inputs=train_inputs,
                train_prior_dist=prior_dist,
                train_labels=labels,
                likelihood=self.likelihood,
            )
        #
        z = self.forward(inputs)
        pred_mean = self.mean_module(
            inputs) + z @ self.prediction_strategy.mean_cache

        if self.full_predictive_covariance:
            precomputed_cache = self.prediction_strategy.covar_cache
            covar_inv_quad_form_root = z @ precomputed_cache

            pred_cov = (
                MatmulLazyTensor(
                    covar_inv_quad_form_root, covar_inv_quad_form_root.transpose(
                        -1, -2)
                )
                .mul(self.likelihood.noise)
                .add_jitter()
            )
        else:
            dim = pred_mean.shape[-1]
            pred_cov = 1e-6 * torch.eye(dim)

        return gpytorch.distributions.MultivariateNormal(pred_mean, pred_cov)

    def forward(self, x):
        """Compute features at location x."""
        z = x @ self.w.transpose(-2, -1) + self.b
        return torch.cat([self.scale * torch.cos(z), self.scale * torch.sin(z)], dim=-1)
