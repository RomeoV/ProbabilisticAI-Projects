import numpy
from scipy.stats import laplace, norm, t
import scipy
import math
import numpy as np
from scipy.special import logsumexp

VARIANCE = 2.0

normal_scale = math.sqrt(VARIANCE)
student_t_df = (2 * VARIANCE) / (VARIANCE - 1)
laplace_scale = VARIANCE / 2

HYPOTHESIS_SPACE = [norm(loc=0.0, scale=math.sqrt(VARIANCE)),
                    laplace(loc=0.0, scale=laplace_scale),
                    t(df=student_t_df)]

PRIOR_PROBS = np.array([0.35, 0.25, 0.4])


def generate_sample(n_samples, seed=None):
    """ data generating process of the Bayesian model """
    random_state = np.random.RandomState(seed)
    hypothesis_idx = np.random.choice(3, p=PRIOR_PROBS)
    dist = HYPOTHESIS_SPACE[hypothesis_idx]
    return dist.rvs(n_samples, random_state=random_state)


""" Solution """

from scipy.special import logsumexp


def log_posterior_probs(x):
    """
    Computes the log posterior probabilities for the three hypotheses, given the data x

    Args:
        x (np.ndarray): one-dimensional numpy array containing the training data
    Returns:
        log_posterior_probs (np.ndarray): a numpy array of size 3, containing the Bayesian log-posterior probabilities
                                          corresponding to the three hypotheses
    """
    assert x.ndim == 1

    log_p = np.zeros(3)

    for (i, (prior, hypothesis)) in enumerate(zip(PRIOR_PROBS, HYPOTHESIS_SPACE)):
        """ Implementation of bayes likelihood eqn
        Given PDFs $p_{X|D}(x|1), p_{X|D}(x|2), p_{X|D}(x|3)$ with data $X$ and distribution $D$,
        we write $p_{D|X} = p_{X|D} * p_D / p_X$ (product rule). In order to prevent numerical underflow, we compute probabilities in log space and
        take the exponential in the end, i.e. we compute $\exp{\log{p_{X|D}}} = \exp{\log{p_{D|X}} + \log{p_X} - \log{p_D}}$.
        Notice now that $p_D$ is given by the problem description. To compute $p_X$, we can use the marginalization rule:
        $p_X = \sum_{j \in D} p_{X|D}(_|j) * p_D(j)$.

        Next, we expand $p_{X|D}$ as $\prod_i p_{X|D}(x_i|D)$, s.t. when taking the logarithm, we get $\log{p_{X|D} = \sum_i \log{p_{X|D}(x_i|D)}}$.
        We do this both for the enumerator and the denominator. Finally, we inject an identity $\exp{\log{\cdot}}$, such that we can use the logexpsum trick.
        More precisely, for $p_X$ we write 
        $$
        \begin{aligned}
        \log{p_X} &= \log{\sum_j \[p_{X|D}(\cdot|j)\]p_D(j) } \\
                  &= \log{\sum \[ \prod_i p_{X|D}(x_i|j) \] p_D(j)} \\
                  &= \log{\sum \[ \exp{\log{ \[ \prod_i p_{X|D}(x_i|j) \]}} \] p_D(j) } \\
                  &= \log{\sum_j \[ \exp \[ \sum_i \log(p_{X|D}(x_i|j)) \] p_D(j)\]}
        \end{aligned}
        $$
        Note that in the last equation, we can use the logsumexp trick for the outer logarithm.
        """
        log_p[i] = hypothesis.logpdf(x).sum() + np.log(prior) \
                 - logsumexp([hypothesis_normalize.logpdf(x).sum() for hypothesis_normalize in HYPOTHESIS_SPACE],
                             b=PRIOR_PROBS)

    assert log_p.shape == (3,)
    return log_p


def posterior_probs(x):
    return np.exp(log_posterior_probs(x))


""" """


def main():
    """ sample from Laplace dist """
    dist = HYPOTHESIS_SPACE[1]
    x = dist.rvs(1000, random_state=28)

    print("Posterior probs for 1 sample from Laplacian")
    p = posterior_probs(x[:1])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior probs for 100 samples from Laplacian")
    p = posterior_probs(x[:50])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior probs for 1000 samples from Laplacian")
    p = posterior_probs(x[:1000])
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))

    print("Posterior for 100 samples from the Bayesian data generating process")
    x = generate_sample(n_samples=100)
    p = posterior_probs(x)
    print("Normal: %.4f , Laplace: %.4f, Student-t: %.4f\n" % tuple(p))


if __name__ == "__main__":
    main()
