"""
Utility functions for computed tomography reconstruction.

This module provides common utility functions for CT reconstruction including
pixel grid generation, line set generation for different sampling geometries,
and reconstruction quality metrics.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import root_mean_squared_error as rmse

def generate_pixel_grid(
    x_max: float,
    y_max: float,
    size: tuple[int, int] = (256, 256)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D pixel grid with coordinates centered at the origin.
    
    This function creates a meshgrid of (x, y) coordinates that represent the 
    spatial locations of pixels in a 2D image.
    
    Parameters
    ----------
    x_max : float
        Maximum extent in the x-direction. The grid will span from -x_max to +x_max.
    y_max : float
        Maximum extent in the y-direction. The grid will span from -y_max to +y_max.
    size : tuple[int, int], default (256, 256)
        Number of pixels in each dimension as (n_x, n_y).
        
    Returns
    -------
    X : np.ndarray
        2D array with shape (size[1], size[0]) containing x-coordinates 
        of pixel centers.
    Y : np.ndarray
        2D array with shape (size[1], size[0]) containing y-coordinates 
        of pixel centers.
        
    Notes
    -----
    The function applies pixel center offset correction by computing:
    - x_offset = x_max / size[0]
    - y_offset = y_max / size[1]
    
    This ensures that pixel coordinates represent the physical center of each
    pixel rather than pixel boundaries.
    
    The resulting coordinate ranges are:
    - X-coordinates: [-x_max + x_offset, x_max - x_offset]
    - Y-coordinates: [-y_max + y_offset, y_max - y_offset]
    """

    # Input validation
    if x_max <= 0 or y_max <= 0:
        raise ValueError("x_max and y_max must be positive")
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("Size dimensions must be positive")

    # Compute offset, so that points are middle points of pixels
    x_off = x_max / size[0]
    y_off = y_max / size[1]

    # Generate grid
    X, Y = np.meshgrid(
        np.linspace(-x_max + x_off, x_max - x_off, size[0]),
        np.linspace(-y_max + y_off, y_max - y_off, size[1])
    )

    return X, Y


def generate_lineset(
    geometry: str,
    r_max: float,
    size: int,
    seed: int = 0,
    a_lim: float = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate line parameters for computed tomography projections.
    
    This function generates sets of line parameters (radial distance and angle)
    for different sampling geometries used in computed tomography reconstruction.
    The lines represent the projection rays through the object being imaged.
    
    Parameters
    ----------
    geometry : str
        Sampling geometry type. Supported options:
        - 'pbg': Parallel beam geometry with optimal sampling based on
          sampling theory for CT reconstruction
        - 'random': Random uniform sampling of line parameters
    r_max : float
        Maximum radial distance for line parameters. Must be positive.
        Defines the extent of the sampling domain: [-r_max, r_max].
    size : int
        Number of projection angles to generate. Must be positive.
        For 'pbg': number of angular samples
        For 'random': total number of random samples
    seed : int, default 0
        Random seed for reproducible random sampling.
        Only used when geometry='random'.
    a_lim : float, default 0
        Angular limit in radians. Must be in range [0, π).
        Angles will be sampled in the range [a_lim, π - a_lim].
        
    Returns
    -------
    r : np.ndarray
        Array of radial distances. Shape depends on geometry:
        - 'pbg': 2D array with shape (size, 2*M+1) where M = floor(size/π)
        - 'random': 1D array with shape (size,)
    a : np.ndarray
        Array of angles in radians. Same shape as r.
        Values are in the range [a_lim, π - a_lim].
        
    Raises
    ------
    ValueError
        If geometry is not 'pbg' or 'random', if r_max is not positive,
        if size is not a positive integer, if a_lim is not in [0, π),
        or if seed is not an integer.
    """

    # Input validation
    if r_max <= 0:
        raise ValueError("r_max must be positive")
    if size <= 0:
        raise ValueError("size must be a positive integer")
    if a_lim < 0 or a_lim >= np.pi:
        raise ValueError("a_lim must be in the range [0, pi)")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    match geometry:
        case 'pbg':
            # Compute number of radial samples via optimal sampling relations
            M = int(np.floor(size / np.pi))

            r, a = np.meshgrid(
                np.linspace(-r_max, r_max, 2 * M + 1),
                np.linspace(a_lim, np.pi - a_lim, size, endpoint=False)
            )
        case 'random':
            # Random parameters via random number generator
            rng = np.random.default_rng(seed)
            r = -r_max + 2 * r_max * rng.random(size)
            a = a_lim + (np.pi - 2 * a_lim) * rng.random(size)
        case _:
            raise ValueError("Unknown geometry type. Use 'pbg' or 'random'.")

    return r, a


def reconstruction_error(
    original: np.ndarray,
    reconstruction: np.ndarray,
    metric: str = 'rmse'
) -> float:
    """
    Calculate reconstruction error using various quality metrics.
    
    This function computes quality metrics to assess the similarity between
    an original image and its reconstruction, commonly used to evaluate
    the performance of reconstruction algorithms in computed tomography
    and other imaging applications.
    
    Parameters
    ----------
    original : np.ndarray
        Original reference image or data array.
        Can be of any dimensionality (1D, 2D, 3D, etc.).
    reconstruction : np.ndarray
        Reconstructed image or data array to compare against the original.
        Must have the same shape as the original array.
    metric : str, default 'rmse'
        Quality metric to compute. Supported options:
        - 'rmse': Root Mean Square Error
        - 'ssim': Structural Similarity Index Measure
        
    Returns
    -------
    float
        Quality metric value:
        - RMSE: Lower values indicate better reconstruction (0 = perfect match)
        - SSIM: Higher values indicate better reconstruction (1 = perfect match)
        
    Raises
    ------
    ValueError
        If original and reconstruction arrays have different shapes,
        or if metric is not 'rmse' or 'ssim'.
    """

    if original.shape != reconstruction.shape:
        raise ValueError("Original and reconstruction must have the same shape.")

    match metric:
        case 'rmse':
            return rmse(original, reconstruction)
        case 'ssim':
            return ssim(
                original,
                reconstruction,
                data_range=reconstruction.max() - reconstruction.min()
            )
        case _:
            raise ValueError("Unknown metric. Use 'rmse' or 'ssim'.")
