from collections import namedtuple
from functools import partial 
from warnings import warn 
from typing import Union, Callable

import numpy as np 

feature_map = lambda f, arr: np.array(list(map(f, arr))).squeeze() 

def polynomial(x: float, degree: int=2) -> np.ndarray: 
    return np.vander(x, degree, increasing=True)

def fourier(x: float, frequencies: list = [1]) -> np.ndarray: 
    sines, cosines = [np.sin(f * x).item() for f in frequencies], \
                     [np.cos(f * x).item() for f in frequencies]
    fourier_basis = list(zip(sines, cosines))
    fourier_basis_with_bias = [1] + [x.item()] + [y for j in fourier_basis for y in j]
    return np.array(fourier_basis_with_bias)

def prepend_ones(arr: np.ndarray) -> np.ndarray: 
    m, n = arr.shape 
    return np.vstack((np.ones_like(arr[:, 0]), arr.T)).T

def _preprocess_features(arr: np.ndarray, **kwargs) -> np.ndarray: 
    return kwargs["feature_map"](arr) if kwargs.get("feature_map", False) else arr

def least_squares(arr: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray: 
    X = _preprocess_features(arr, **kwargs) 
    return np.linalg.solve(arr.T @ arr, arr.T.dot(targets))

def bayesian_regression(arr: np.ndarray, targets: np.ndarray, **kwargs) -> callable: 
    X = _preprocess_features(arr, **kwargs) 

    noise_scale = kwargs.get("noise_scale", 1e-2) 
    m, n = arr.shape
    prior_covariance = np.eye(n)
    posterior_covariance = np.linalg.inv((X.T @ X)/noise_scale + np.linalg.inv(prior_covariance))
    return posterior_covariance @ X.T @ targets * 1/noise_scale
