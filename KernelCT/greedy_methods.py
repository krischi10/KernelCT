"""
Greedy methods for kernel-based CT reconstruction.

This module provides greedy algorithms for selecting kernel centers in
kernel-based CT reconstruction methods, including geometric and beta-greedy
approaches for optimal sampling point selection.
"""

import numpy as np
from weighted_kernels import WeightedGaussian


def geometric_greedy(
    dist: np.ndarray,
    greedy_method: str,
    r: np.ndarray,
    a: np.ndarray,
    r_center: float,
    a_center: float,
    kernel: WeightedGaussian = None,
    s: float = 1.2
) -> np.ndarray:
    """
    Perform geometric greedy selection for new interpolation point.
    
    Parameters
    ----------
    dist : np.ndarray
        Current distance array
    greedy_method : str
        Method type: 'geo_dual_space', 'geo_parameter_space', 'geo_periodic', 'geo_sphere'
    r : np.ndarray
        Radial coordinates
    a : np.ndarray
        Angular coordinates
    r_center : float
        Center radial coordinate
    a_center : float
        Center angular coordinate
    kernel : WeightedGaussian, optional
        Kernel instance for dual space methods
    s : float, default 1.2
        Scaling parameter for sphere method
        
    Returns
    -------
    tuple[np.ndarray, int]
        Updated distance array and selected index
    """

    # Input validation
    if not isinstance(dist, np.ndarray):
        raise TypeError("dist must be a numpy array.")
    if not isinstance(greedy_method, str):
        raise TypeError("greedy_method must be a string.")
    if not isinstance(r, np.ndarray):
        raise TypeError("r must be a numpy array.")
    if not isinstance(a, np.ndarray):
        raise TypeError("a must be a numpy array.")
    if not isinstance(r_center, (float, int)):
        raise TypeError("r_center must be a float or int.")
    if not isinstance(a_center, (float, int)):
        raise TypeError("a_center must be a float or int.")
    if greedy_method == 'geo_dual_space' and not isinstance(kernel, WeightedGaussian):
        raise TypeError("kernel must be an instance of WeightedGaussian.")
    if greedy_method == 'geo_sphere' and not isinstance(s, (float, int)):
        raise TypeError("s must be a float or int.")


    match greedy_method:
        case 'geo_dual_space':
            dist_new = (kernel.norm_radon_functional(r) 
                        - 2 * kernel.gram_matrix(r, a, r_center, a_center)
                        + kernel.norm_radon_functional(r_center))

        case 'geo_parameter_space':
            dist_new = (r - r_center)**2 + (a - a_center)**2

        case 'geo_periodic':
            dist_new = np.min([
                (r - r_center)**2 + (a - a_center)**2,
                (r + r_center)**2 + (a - a_center + np.pi)**2,
                (r + r_center)**2 + (a - a_center - np.pi)**2
            ], axis=0)

        case 'geo_sphere':
            inner_product = np.clip(
                ((1 / np.sqrt(1 + r**2 * s**2)) * (1 / np.sqrt(1 + r_center**2 * s**2))
                * (np.cos(a - a_center) + r * r_center * s**2)),
                -1,
                1
            )  # avoid leaving the domain of arccos via truncation
            dist_new = np.minimum(np.arccos(inner_product), np.arccos(-inner_product))
        case _:
            raise ValueError("greedy_method must be one of: 'geo_dual_space', "
                             "'geo_parameter_space', 'geo_periodic', 'geo_sphere'.")

    dist = np.minimum(dist, dist_new)
    index = np.argmax(dist)

    return dist, index


def beta_greedy(
    pwr_func_vals: np.ndarray,
    res: np.ndarray,
    beta: float | int | str = 0.5
) -> int:
    """
    Perform beta-greedy selection for new interpolation point.
    
    Parameters
    ----------
    pwr_func_vals : np.ndarray
        Power function values at candidate points
    res : np.ndarray
        Residual values at candidate points
    beta : float or int or str, default 0.5
        Beta parameter for greedy selection
        
    Returns
    -------
    int
        Selected index for new kernel center
        
    Raises
    ------
    TypeError
        If input types are incorrect
    ValueError
        If beta is negative
    """

    # Input validation
    if not isinstance(pwr_func_vals, np.ndarray):
        raise TypeError("pwr_func_vals must be a numpy array.")
    if not isinstance(res, np.ndarray):
        raise TypeError("res must be a numpy array.")
    if not isinstance(beta, (float, str)):
        raise TypeError("beta must be a float or 'inf'.")
    if isinstance(beta, (float, int)) and beta < 0:
        raise ValueError("beta must be non-negative or 'inf'.")

    indices = list(np.where(pwr_func_vals >= 1e-20)[0])  # avoid division by zero

    if beta == 'inf':
        i = np.argmax(np.absolute(res[indices]) / np.sqrt(pwr_func_vals[indices]))
    else:
        i = np.argmax(np.absolute(res[indices])**beta 
                      * np.sqrt(pwr_func_vals[indices])**(1-beta))

    return indices[i]
