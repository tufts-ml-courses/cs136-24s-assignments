'''
Defines a function that computes the likelihood of data under a GMM.

Needs to be auto-differentiable via autograd package.

Usage
-----
Import the `calc_neg_log_lik` function to use it

Examples
--------
## Setup: Create useful parameters
>>> import numpy as np
>>> np.set_printoptions(precision=3, suppress=1)
>>> K = 3
>>> D = 2
>>> log_pi_K = np.log([1./3, 1./3, 1./3]);
>>> stddev_KD = np.ones((K, D))
>>> mu_KD = np.zeros((K, D))
>>> mu_KD[0,:] = -1.0
>>> mu_KD[-1,:] = +1.0
>>> mu_KD
array([[-1., -1.],
       [ 0.,  0.],
       [ 1.,  1.]])

## Neg. likelihood of empty dataset should be zero
>>> empty_ND = np.zeros((0,D))
>>> calc_neg_log_lik(empty_ND, log_pi_K, mu_KD, stddev_KD)
-0.0

## Neg. likelihood of dataset of all zeros should be large
>>> N = 4
>>> allzero_x_ND = np.zeros((N,D))
>>> print("%.3f" % calc_neg_log_lik(allzero_x_ND, log_pi_K, mu_KD, stddev_KD))
9.540

## Neg. likelihood of bigger dataset should be even larger
>>> N = 8
>>> bigzero_x_ND = np.zeros((N,D))
>>> print("%.3f" % calc_neg_log_lik(bigzero_x_ND, log_pi_K, mu_KD, stddev_KD))
19.080

## Gradient of neg log likelihood should be easy using autograd
>>> import autograd
>>> calc_grad_of_log_pi_K = autograd.grad(calc_neg_log_lik, argnum=1)
>>> calc_grad_of_log_pi_K(bigzero_x_ND, log_pi_K, mu_KD, stddev_KD)
array([-1.696, -4.609, -1.696])

>>> calc_grad_of_mu_KD = autograd.grad(calc_neg_log_lik, argnum=2)
>>> calc_grad_of_mu_KD(bigzero_x_ND, log_pi_K, mu_KD, stddev_KD)
array([[-1.696, -1.696],
       [ 0.   ,  0.   ],
       [ 1.696,  1.696]])
'''

import autograd.numpy as ag_np
import autograd.scipy.stats as ag_stats
from autograd.scipy.special import logsumexp

def calc_neg_log_lik(x_ND, log_pi_K, mu_KD, stddev_KD):
    ''' Calculate negative log likelihood of observations under provided GMM parameters

    Negative log likelihood is $-1 * log p(x)$
    
    where the log likelihood $\log p(x)$ of a dataset $x = \{x_n\}_{n=1}^N$ is defined by:
        \begin{align}
        log p(x) = \sum_{n=1}^N \log GMMPDF(x_n | \pi, \mu, \sigma)
        \end{align}

    NB: Here, the likelihood is the "marginal" likelihood or "incomplete" likelihood.

    Args
    ----
    x_ND : 2D array, shape (N, D)
        Observed data array.
        Each row is a feature vector of size D (num. feature dimensions).
    log_pi_K : 1D array, shape (K,)
        GMM parmaeter: Log of mixture weights
        Must satisfy logsumexp(log_pi_K) == 0.0 (which means sum(exp(log_pi_K)) == 1.0)
    mu_KD : 2D array, shape (K, D)
        GMM parameter: Means of all components
        The k-th row is the mean vector for the k-th component
    stddev_KD : 2D array, shape (K, D)
        GMM parameter: Standard Deviations of all components
        The k-th row is the stddev vector for the k-th component

    Returns
    -------
    neg_log_lik : float
        Negative log likelihood of provided dataset
    '''
    assert x_ND.ndim == 2
    N, D = x_ND.shape
    assert D == mu_KD.shape[1]
    assert D == stddev_KD.shape[1]
    K = mu_KD.shape[0]

    ## TODO write code to compute the negative log likelihood
    neg_log_lik_placeholder = ag_np.sum(mu_KD)
    ## Make sure this code is differentiable with autograd
    ## This means:
    ## * use ag_np instead of np for any numpy routines
    ## * don't use any other libraries except what is already imported here
    return neg_log_lik_placeholder