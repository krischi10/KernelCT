import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import convolve2d


def low_pass_filter(radon_vals: np.ndarray, filter_type: str, L: float, filter_width: float) -> np.ndarray:
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

    # Indices for radial variable discretization
    t = np.arange(-filter_width, filter_width + 1)

    match filter_type:
        case 'ram_lak':
            conv_filter = (L**2 / (2 * np.pi)) * (2 * np.sinc(t) - np.sinc(t/2)**2)
        case 'shepp_logan':
            conv_filter = 4 * L**2 / (np.pi**3 * (1 - 4 * t**2))
        case _:
            raise ValueError("Unknown filter type. Use 'ram_lak' or 'shepp_logan'.")

    return convolve2d(radon_vals, conv_filter[np.newaxis, :], mode='valid')


def row_wise_splines(x_eval, x_data, vals, spline_type='linear'):
    """
    
    """

    spline_eval = np.zeros_like(x_eval)

    if spline_type == 'linear':
        for i in range(x_eval.shape[0]):
            spline_eval[i, :] = np.interp(x_eval[i, :], x_data, vals[i, :])

    elif spline_type == 'cubic':
        for i in range(x_eval.shape[0]):         
            cs = CubicSpline(x_data, vals[i, :])
            spline_eval[i, :] = cs(x_eval[i, :])

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

    return (1 / (2 * np.shape(h)[0])) * np.sum(h, axis = 0)