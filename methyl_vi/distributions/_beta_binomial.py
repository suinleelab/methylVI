from typing import Optional

import torch
from pyro.distributions import BetaBinomial as BetaBinomialDistribution
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all


class BetaBinomial(BetaBinomialDistribution):
    r"""Beta binomial distribution.

    One of the following parameterizations must be provided:

    (1), (`alpha`, `beta`, `total_counts`) where alpha and beta are the shape parameters of
    the beta distribution and total_counts is the number of trials. (2), (`mu`, `gamma`,
    `total_counts`) parameterization, which is the one used by methylVI. These
    parameters respectively control the mean and dispersion of the distribution.

    Parameters
    ----------
    total_count
        Number of trials.
    alpha
        First shape parameter of the beta distribution.
    beta
        Second shape parameter of the beta distribution.
    mu
        Mean of the distribution.
    gamma
        Dispersion.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None,
        validate_args: bool = False,
    ):
        self._eps = 1e-8
        if (mu is None) == (alpha is None):
            raise ValueError(
                "Please use one of the two possible parameterizations. Refer to the documentation for more information."
            )

        using_param_1 = alpha is not None and beta is not None

        if using_param_1:
            alpha, beta = broadcast_all(alpha, beta)
        else:
            mu, gamma = broadcast_all(mu, gamma)
            mu = torch.clamp(mu, min=self._eps)
            gamma = torch.clamp(gamma, min=self._eps)  # To avoid divide by 0 issues

            alpha = mu * (1 - gamma) / gamma
            beta = (1 - mu) * (1 - gamma) / gamma

            # Due to numerical stability issues, sometimes alpha or beta will end up
            # as exact zeros, so we clamp at a small positive number epsilon
            alpha = torch.clamp(alpha, min=self._eps)
            beta = torch.clamp(beta, min=self._eps)

        self.alpha = alpha
        self.beta = beta

        super().__init__(
            concentration1=alpha,
            concentration0=beta,
            total_count=total_count,
            validate_args=validate_args,
        )
