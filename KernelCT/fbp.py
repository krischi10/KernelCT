"""
Filtered Back Projection (FBP) reconstruction algorithms for computed tomography.

This module provides functions for applying reconstruction filters, 
interpolation, and back projection operations commonly used in CT reconstruction.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import convolve2d


def low_pass_filter(
    radon_vals: np.ndarray, 
    filter_type: str, 
    L: float, 
    filter_width: float
) -> np.ndarray:
    """
    Apply a low-pass filter to Radon transform data for filtered back projection.
    
    This function generates and applies reconstruction filters commonly used in 
    computed tomography (CT) for the filtered back projection algorithm. The filters
    help reduce noise and artifacts in the reconstructed image by attenuating
    high-frequency components in the projection data.
    
    Parameters
    ----------
    radon_vals : np.ndarray
        Radon transform data with shape (n_angles, n_radii).
    filter_type : str
        Type of reconstruction filter to apply. Supported options:
        - 'ram_lak': Ram-Lak filter
        - 'shepp_logan': Shepp-Logan filter
    L : float
        Bandwidth of the filter. Controls the frequency response and
        determines the cutoff characteristics of the reconstruction filter.
    filter_width : float
        Half-width of the filter kernel. The total filter length will be
        2 * filter_width + 1.
        
    Returns
    -------
    np.ndarray
        Filtered Radon data with shape (n_angles, n_pixels - 2*filter_width).
        The convolution with 'valid' mode reduces the output size.
        
    Raises
    ------
    ValueError
        If filter_type is not 'ram_lak' or 'shepp_logan'.
    """

    # Input validation
    if filter_type not in ['ram_lak', 'shepp_logan']:
        raise ValueError("filter_type must be 'ram_lak' or 'shepp_logan'")
    if L <= 0:
        raise ValueError("L must be positive")
    if filter_width <= 0:
        raise ValueError("filter_width must be positive")

    # Indices for radial variable discretization
    t = np.arange(-filter_width, filter_width + 1)

    match filter_type:
        case 'ram_lak':
            conv_filter = (L**2 / (2 * np.pi)) * (2 * np.sinc(t) - np.sinc(t / 2)**2)
        case 'shepp_logan':
            conv_filter = 4 * L**2 / (np.pi**3 * (1 - 4 * t**2))
        case _:
            raise ValueError("Unknown filter type. Use 'ram_lak' or 'shepp_logan'.")

    return convolve2d(radon_vals, conv_filter[np.newaxis, :], mode='valid')


def row_wise_splines(
    x_eval: np.ndarray, 
    x_data: np.ndarray, 
    vals: np.ndarray, 
    spline_type: str = 'linear'
) -> np.ndarray:
    """
    Perform row-wise spline interpolation on 2D data arrays.
    
    This function interpolates each row of a values array independently using
    spline interpolation. Each row is interpolated from the fixed array of data
    points to new evaluation points, enabling batch processing of
    multiple 1D interpolation problems.
    
    Parameters
    ----------
    x_eval : np.ndarray
        2D array of evaluation points with shape (n_rows, n_eval_points).
        Each row contains the points where interpolation will be evaluated
        for the corresponding row in vals.
    x_data : np.ndarray
        1D array of data points with shape (n_data_points,).
        The x-coordinates of the known data points used for interpolation.
        Must be in ascending order for proper interpolation.
    vals : np.ndarray
        2D array of values with shape (n_rows, n_data_points).
        Each row contains the y-values corresponding to x_data points
        that will be interpolated to x_eval points.
    spline_type : str, default 'linear'
        Type of interpolation to perform. Supported options:
        - 'linear': Linear interpolation between adjacent points
        - 'cubic': Cubic spline interpolation for smooth curves
        
    Returns
    -------
    np.ndarray
        2D array with shape (n_rows, n_eval_points) containing the
        interpolated values at x_eval points for each row.
        
    Raises
    ------
    ValueError
        If spline_type is not 'linear' or 'cubic'.
    """

    # Input validation
    if spline_type not in ['linear', 'cubic']:
        raise ValueError("spline_type must be 'linear' or 'cubic'")
    if x_eval.shape[0] != vals.shape[0]:
        raise ValueError("x_eval and vals must have the same number of rows")
    if vals.shape[1] != len(x_data):
        raise ValueError("vals number of columns must match x_data length")

    spline_eval = np.zeros_like(x_eval)

    match spline_type:
        case 'linear':
            for i in range(x_eval.shape[0]):
                spline_eval[i, :] = np.interp(x_eval[i, :], x_data, vals[i, :])
        case 'cubic':
            for i in range(x_eval.shape[0]):
                cs = CubicSpline(x_data, vals[i, :])
                spline_eval[i, :] = cs(x_eval[i, :])
        case _:
            raise ValueError("Unknown spline type. Use 'linear' or 'cubic'.")

    return spline_eval


def back_projection(h: np.ndarray) -> np.ndarray:
    """
    Perform back projection step of the filtered back projection (FBP) algorithm.
    
    This function computes the back projection by summing the filtered projection data
    across all angles and normalizing by the number of projections. The back projection
    step reconstructs the spatial domain image from the Radon transform data.
    
    Parameters
    ----------
    h : np.ndarray
        Filtered projection data with shape (n_angles, n_pixels).
        Each row represents a filtered projection at a specific angle.
        
    Returns
    -------
    np.ndarray
        Back projected image with shape (n_pixels,).
        The reconstructed spatial domain representation.
    """

    # Input validation
    if h.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if h.shape[0] == 0 or h.shape[1] == 0:
        raise ValueError("Input array cannot be empty")

    return (1 / (2 * h.shape[0])) * np.sum(h, axis=0)