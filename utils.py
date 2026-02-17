# utils.py
import numpy as np
import pandas as pd
from scipy.stats import truncnorm, beta, multivariate_normal

def set_seed(seed):
    np.random.seed(seed)
    import random
    random.seed(seed)

def truncated_normal(mean, std, low, high, size=1):
    a, b = (low - mean) / std, (high - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

def beta_dist(alpha, beta, size=1):
    return np.random.beta(alpha, beta, size=size)

def multivariate_normal_sample(mean, cov, size=1):
    return np.random.multivariate_normal(mean, cov, size=size)

def add_noise(value, std, min_val=None, max_val=None):
    noise = np.random.normal(0, std, size=len(value))
    new_val = value + noise
    if min_val is not None:
        new_val = np.maximum(new_val, min_val)
    if max_val is not None:
        new_val = np.minimum(new_val, max_val)
    return new_val