import pathing 
from functools import partial 
import numpy as np 
import matplotlib.pyplot as plt 
import numpy.random as npr 

from regression import least_squares, feature_map, polynomial, fourier, bayesian_regression

if __name__=="__main__": 
    # visuals 
    span = (-5, 5)
    resolution = 100 
    dom = np.linspace(*span, num=resolution).reshape(-1, 1)

    # form a synthetic regression problem 
    n_data = 15 
    noise_scale = 1e-2
    f = lambda arr: np.tanh(arr) + np.sin(arr)
    arr = np.linspace(*span, num=n_data).reshape(-1, 1)
    targets = f(arr) + npr.randn(n_data).reshape(-1, 1) * noise_scale 

    plt.figure()

    for degree in range(1, 5): 
        _feature_map = partial(feature_map, partial(polynomial, degree=degree))
        parameters = least_squares(arr, targets, feature_map=_feature_map) 
        polynomial_predictor = lambda x: x.dot(parameters)
        plt.plot(dom, polynomial_predictor(dom), label=f"polynomial {degree}")

    for bandwidth in range(2, 5): 
        _feature_map = partial(feature_map, partial(fourier, frequencies=list(range(1, bandwidth))))
        parameters = least_squares(arr, targets, feature_map=_feature_map)
        fourier_predictor = lambda x: x.dot(parameters) 
        plt.plot(dom, fourier_predictor(dom), label=f"fourier (bandwidth={bandwidth})")

    plt.scatter(arr, targets, marker='x')
    plt.plot(dom, f(dom), label="true function")
    plt.legend() 
    plt.show() 
