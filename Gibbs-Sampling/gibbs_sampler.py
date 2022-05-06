from time import time
import os
import numpy as np
from numpy.random import binomial, normal, beta, multinomial
import scipy.stats as st
from scipy.stats import invgamma, norm, dirichlet, multivariate_normal
from distcan import InverseGamma
import pandas as pd
from multiprocessing import Pool, cpu_count
from sklearn.metrics import rand_score, adjusted_rand_score, silhouette_score

def data_gen(mu, sigmas, phi, n):
    """
    Generates samples from Mixture of K Gaussian Distributions
    """
    y = []
    class_list = []
    for i in range(n):
        ind = multinomial(1, phi)
        class_list.append(np.argmax(ind))
        
        for j, val in enumerate(ind):
            if val == 1:
                y.append(norm(mu[j], sigmas[j]).rvs())
            else:
                next
    return np.array(y), np.array(class_list)

def update_pi(alpha_vec, z_vec):
    """
    Sample from Posterior Conditional for pi
    """
    assert len(z_vec) == len(alpha_vec), "Number of distributions must equal number of parameters"
    return dirichlet(z_vec + alpha_vec).rvs()

def update_mu(y, z_mat, sigma_vec):
    """
    Sample from Posterior Conditional for mu
    """
    mu_vec = []
    n_j =  np.sum(z_mat, axis=0)
    for j in range(len(sigma_vec)):
        sigma_vec[j] = sigma_vec[j] / (n_j[j] + 1)
        mu_vec.append(np.sum(y * z_mat[:,j]) / (n_j[j] + 1))
    
    cov = np.diag(sigma_vec)
    return multivariate_normal(mu_vec, cov).rvs()

def update_sigma(data, z_mat, mu):
    """
    Sample from Posterior Conditional for sigma
    """
    n_j = np.sum(z_mat, axis=0)
    alpha = (0.5 * n_j) + 1
    beta = []
    for j in range(len(mu)):
        y = data * z_mat[:,j]
        y = y[y != 0]
        beta.append((0.5 * np.square(y - mu[j]).sum()) + 1)
    return InverseGamma(alpha, beta).rvs()

def update_z(data: list, mu, sigma, pi):
    """
    Sample from latent variable Z according to likelihoods for class assignment
    """
    a = np.empty((len(data), len(mu)))
    out = np.empty((len(data), len(mu)))
    for j in range(len(mu)):
        a[:,j] = norm(mu[j], np.sqrt(sigma[j])).pdf(data) * pi[0,j]
    
    pi_i = a / np.sum(a, axis=1)[:,None]
    for i in range(len(data)):
        out[i,] = multinomial(1, pi_i[i,:])
    return out

def gibbs(data, iters, burnin, k, seed=None):
    """
    Run Gibb's Sampling for Mixture of 2 Gaussians. Initial States are sample from Priors
    """
    if seed is not None:
        np.random.seed(seed)

    try:
        # Set initial guesses based on priors
        alpha = np.repeat(1, k)
        mu = normal(0, 1, size=k)
        pi = dirichlet(alpha).rvs()
        sigma = InverseGamma(1,1).rvs(size=k)
        out = np.empty((iters, k*3))

        for i in range(iters):
            # Update Parameters according to conditional posterior distributions
            z_mat = update_z(data, mu, sigma, pi)
            pi = update_pi(alpha, np.sum(z_mat, axis=0))
            mu = update_mu(data, z_mat, sigma)
            sigma = update_sigma(data, z_mat, mu)

            # Store Values to monitor trace
            out[i, 0*k:1*k] = mu
            out[i, 1*k:2*k] = np.sqrt(sigma)
            out[i, 2*k:3*k] = pi[0,:]
        
        return [mu, np.sqrt(sigma), pi, np.argmax(z_mat, axis=1), out[burnin:,:]]
    
    except Exception as e: 
        print(e)
        return [None, e]

def make_param_dict(trace):
    k = int(trace.shape[1]/3)
    params_dict = {}
    for j in range(k):
        params_dict.update(
            {
                f"mu{j}": np.round(np.mean(trace[:,j]),2),
                f"sigma{j}": np.round(np.mean(trace[:,(j+k)]),2),
                f"pi{j}": np.round(np.mean(trace[:,(j+2*k)]),2)
            }
        )

    return params_dict

def pred_label(data, params_dict):
    k = int(len(params_dict)/3)
    z_probmat = []
    for i in range(len(data)):
        prob_z = []
        for j in range(k):
            prob_z.append(norm.pdf(data[i], loc=params_dict[f"mu{j}"], scale=params_dict[f"sigma{j}"]))
        z_probmat.append(prob_z)

    z_est_mean = np.argmax(z_probmat, axis=1)

    return z_est_mean

def score(y, true, pred):
    rs = rand_score(true, pred)
    ars = adjusted_rand_score(true, pred)
    ss = silhouette_score(y.reshape(-1,1), pred.reshape(-1,1))

    print(f"Rand Index = {rs:.3f}")
    print(f"Adjusted Rand Index = {ars:.3f}")
    print(f"Silhouette Score = {ss:.3f}")

    return [rs, ars, ss]

def multigibbs_gibbs(y, k, N_itter, burnin, n_init=8):
    pool = Pool()
    inputs = []
    seeds = np.random.randint(1000, size=n_init)
    
    for i in range(1,n_init):
        inputs.append((y, N_itter, burnin, k, seeds[i-1]))

    with Pool() as pool:
        res = pool.starmap(gibbs, inputs)

    # Calculating Silhouette score to find the best gibbs sampler
    data = y.reshape(-1,1)
    ss_list = []
    for i in range(1,n_init):
        if res[i-1][0] is None:
            ss = -1
        else:
            ss = silhouette_score(data, res[i-1][3].reshape(-1,1))
            
        ss_list.append(ss)
        
    best_model = res[np.argmax(ss_list)]

    return best_model


def unknown_k_gibbs(y, N_itter, burnin, k_max = 10):
    pool = Pool()
    inputs = []

    for k in range(1,k_max):
        inputs.append((y, N_itter, burnin, k))

    with Pool() as pool:
        res = pool.starmap(gibbs, inputs)

    # Calculating Silhouette score to find the best gibbs sampler
    data = y.reshape(-1,1)
    ss_list = []
    for i in range(1,k_max):
        if res[i-1][0] is None:
            ss = -1
        else:
            ss = silhouette_score(data, res[i-1][3].reshape(-1,1))
                        
        ss_list.append(ss)

        best_model = res[np.argmax(ss_list)]

    return best_model, ss_list, res

