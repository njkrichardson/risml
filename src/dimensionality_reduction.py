import numpy as np 
from numpy import cov 
from numpy.linalg import eig
import numpy.random as npr 

def center(arr: np.ndarray, sample_dimension: int=0) -> np.ndarray: 
    """Center an array along an axis by reducing each element by the 
    sample mean. 

    Parameters
    ----------
    arr : np.ndarray
        array to be centered. 
    sample_dimension : int
        dimension along which to compute the sample mean. 

    Returns
    -------
    np.ndarray
        centered array. 
    """
    return arr - arr.mean(axis=sample_dimension)

def pca(X: np.ndarray, rank: int) -> np.ndarray: 
    """Linearly project the elements of an array X onto a dimension `rank`
    utilizing principal components analysis (PCA); the projection matrix 
    is construced using the eigenvectors of the covariance matrix of X. 

    Parameters
    ----------
    X : np.ndarray
        Data matrix. 
    rank : int
        Dimension of the subspace to project elements of X onto. 

    Returns
    -------
    np.ndarray
        Latent/compressed/projections of elements of X, each of dimension `rank`. 
    """
    X = X if X.shape[0] > X.shape[1] else X.T 
    X = center(X) 
    eigenvalues, eigenvectors = eig(cov(X.T))
    idxs = np.argsort(eigenvalues)[::-1] 
    F = eigenvectors[:, idxs][:, :rank].T
    return (F @ X.T).T

