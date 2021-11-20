from copy import deepcopy 
from typing import Callable, Tuple

import jax.numpy as np 
from jax import grad, jacfwd, jacrev
from numpy.linalg import solve, norm 

def gradient(f: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], np.ndarray]: 
    return grad(f) 

def jacobian(f: Callable[[np.ndarray], float], forward: bool=True) -> Callable[[np.ndarray], np.ndarray]: 
    return jacfwd(f) if forward else jacrev(f) 

def hessian(f: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], np.ndarray]: 
    return jacfwd(jacrev(f))

def newton(f: Callable[[np.ndarray], float], x_initial: np.ndarray, **kwargs) -> np.ndarray: 
    m, n = x_initial.size, f(x_initial).size
    x = deepcopy(x_initial)
    update = _newton_update(f, (m, n))

    for _ in range(kwargs.get("max_iterations", int(1e2))): 
        dx = update(x) 
        x -= dx
        if norm(dx) < kwargs.get("convergence_tol", float(1e-3)): 
            break 
    return x 

def _newton_update(f: Callable[[np.ndarray], float], shape: Tuple[int, int]) -> Callable[[np.ndarray], np.ndarray]: 
    m, n = shape 

    if m > 1: 
        jacobian_f = jacobian(f); 
        return lambda x: (solve(jacobian_f(x) @ jacobian_f(x), jacobian(x).T)) @ f(x) \
                if m > n else lambda x: solve(jacobian_f(x), -f(x))
    else: 
        gradient_f = grad(f); 
        return lambda x: f(x) / gradient_f(x) 

def gradient_descent(f: Callable[[np.ndarray], float], x_initial: np.ndarray, **kwargs) -> np.ndarray: 
    x = deepcopy(x_initial)
    gradient_f = gradient(f)
    for i in range(kwargs.get("max_iterations", int(1e2))): 
        x -= kwargs.get("step_size", 1e-2) * gradient_f(x)
        if x <= kwargs.get("convergence_tol", 1e-3): 
            break 
    return x 

def conjugate_gradient(A: np.ndarray, b: np.ndarray) -> np.ndarray: 
    n, _ = A.shape
    x = np.zeros(n, dtype=np.float32)
    residual = b - A.dot(x)
    if norm(residual) < kwargs.get("convergence_tol", 1e-2): 
        return x
    direction = residual 
    for _ in range(n-1): 
        previous_residual = deepcopy(residual)
        alpha = residual.dot(residual) / direction.dot(A.dot(direction))
        x += alpha * direction 
        residual -= alpha * A.dot(direction)
        if norm(residual) < kwargs.get("convergence_tol", 1e-2): 
            break
        beta = residual.dot(residual) / previous_residual.dot(previous_residual)
        direction = residual + beta * direction 
    return x 
