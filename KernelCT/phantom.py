"""
Analytical phantom functions and Radon transforms for CT reconstruction testing.

Provides standard test phantoms (bulls_eye, crescent_shaped, shepp_logan, smooth_phantom)
and their corresponding analytical Radon transforms for algorithm validation.
"""

import numpy as np
from scipy.special import gamma


# Phantom parameter constants
SHEPP_LOGAN_PARAMS = {
    'x_centers': np.array([0, 0, 0.22, -0.22, 0, 0, 0, -0.08, 0, 0.06]),
    'y_centers': np.array([0, -0.0184, 0, 0, 0.35, 0.1, -0.1, 
                          -0.605, -0.605, -0.605]),
    'x_widths': np.array([0.69, 0.6624, 0.11, 0.16, 0.21, 0.046, 
                         0.046, 0.046, 0.023, 0.023]),
    'y_widths': np.array([0.92, 0.874, 0.31, 0.41, 0.25, 0.046, 
                         0.046, 0.023, 0.023, 0.046]),
    'rotations': np.array([0, 0, -np.pi/10, np.pi/10, 0, 0, 0, 0, 0, 0]),
    'vals': np.array([1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
}

SMOOTH_PHANTOM_PARAMS = {
    'x_centers': np.array([0.22, -0.22, 0]),
    'y_centers': np.array([0, 0, 0.2]),
    'x_widths': np.array([0.51, 0.51, 0.5]),
    'y_widths': np.array([0.31, 0.36, 0.8]),
    'rotations': np.array([(2/5)*np.pi, (3/5)*np.pi, np.pi/2]),
    'vals': np.array([1, -3/2, 3/2]),
    'alpha': 3
}


def eval_phantom(X: np.ndarray, Y: np.ndarray, 
                 phantom_type: str = 'bulls_eye') -> np.ndarray:
    """
    Evaluate standard CT phantom functions at coordinate points.
    
    Parameters
    ----------
    X : np.ndarray
        X-coordinates of evaluation points
    Y : np.ndarray
        Y-coordinates of evaluation points (same shape as X)
    phantom_type : str, default 'bulls_eye'
        Type of phantom to evaluate:
        - 'bulls_eye': Concentric circles with alternating densities
        - 'crescent_shaped': Two overlapping circles
        - 'shepp_logan': Modified Shepp-Logan phantom (10 ellipses)
        - 'smooth_phantom': Smooth phantom with differentiable boundaries
        
    Returns
    -------
    np.ndarray
        Phantom attenuation values at the given coordinates
        
    Raises
    ------
    TypeError
        If inputs are not of correct type
    ValueError
        If X and Y have different shapes, are empty, or phantom_type is 
        invalid
    """

    if not isinstance(phantom_type, str):
        raise TypeError("phantom_type must be a string")
    
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array")
    
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")
    if X.size == 0:
        raise ValueError("X and Y cannot be empty arrays")

    match phantom_type:
        case 'bulls_eye':
            phantom_values = ((X**2 + Y**2 <= 9/16) - (3/4) *
                             (X**2 + Y**2 <= 1/4) +
                             (1/4) * (X**2 + Y**2 <= 1/16))

        case 'crescent_shaped':
            phantom_values = ((X**2 + Y**2 <= 1/4) -
                             (1/2) * (((X - (1/8))**2 + Y**2) <= (9/64)))

        case 'shepp_logan':
            # Parameters for ellipses
            x_centers = SHEPP_LOGAN_PARAMS['x_centers']
            y_centers = SHEPP_LOGAN_PARAMS['y_centers']
            x_widths = SHEPP_LOGAN_PARAMS['x_widths']
            y_widths = SHEPP_LOGAN_PARAMS['y_widths']
            rotations = SHEPP_LOGAN_PARAMS['rotations']
            vals = SHEPP_LOGAN_PARAMS['vals']

            # Initialize point eval loop over ellipses
            phantom_values = np.zeros_like(X)

            for i in range(len(x_centers)):
                x_rot = ((X - x_centers[i]) * np.cos(rotations[i]) + 
                        (Y - y_centers[i]) * np.sin(rotations[i]))
                y_rot = (-(X - x_centers[i]) * np.sin(rotations[i]) + 
                        (Y - y_centers[i]) * np.cos(rotations[i]))
                phantom_values = phantom_values + vals[i] * (
                    (x_rot**2 / x_widths[i]**2 + 
                     y_rot**2 / y_widths[i]**2) <= 1
                )

        case 'smooth_phantom':
            # smoothness parameter
            alpha = SMOOTH_PHANTOM_PARAMS['alpha']

            # Parameters for ellipses
            x_centers = SMOOTH_PHANTOM_PARAMS['x_centers']
            y_centers = SMOOTH_PHANTOM_PARAMS['y_centers']
            x_widths = SMOOTH_PHANTOM_PARAMS['x_widths']
            y_widths = SMOOTH_PHANTOM_PARAMS['y_widths']
            rotations = SMOOTH_PHANTOM_PARAMS['rotations']
            vals = SMOOTH_PHANTOM_PARAMS['vals']

            # Initialize point eval and Radon data + loop over ellipses
            phantom_values = np.zeros_like(X)

            for i in range(len(x_centers)):
                x_rot = ((X - x_centers[i]) * np.cos(rotations[i]) + 
                        (Y - y_centers[i]) * np.sin(rotations[i]))
                y_rot = (-(X - x_centers[i]) * np.sin(rotations[i]) + 
                        (Y - y_centers[i]) * np.cos(rotations[i]))
                
                phantom_values = phantom_values + vals[i] * np.maximum(
                    1 - (x_rot**2 / x_widths[i]**2 + 
                         y_rot**2 / y_widths[i]**2),
                    0)**alpha

        case _:
            raise ValueError(f"Unknown phantom type: {phantom_type}")

    return phantom_values

def eval_phantom_radon(r: np.ndarray, a: np.ndarray, 
                       phantom_type: str = 'bulls_eye') -> np.ndarray:
    """
    Compute analytical Radon transform for standard CT phantoms.
    
    Parameters
    ----------
    r : np.ndarray
        Radial coordinates for line parameters (signed distance from origin)
    a : np.ndarray
        Angular coordinates for line parameters in radians (same shape as r)
    phantom_type : str, default 'bulls_eye'
        Type of phantom:
        - 'bulls_eye': Analytical transform using chord length formula
        - 'crescent_shaped': Transform of two overlapping circles
        - 'shepp_logan': Transform of modified Shepp-Logan phantom
        - 'smooth_phantom': Transform with gamma function expressions
        
    Returns
    -------
    np.ndarray
        Radon transform values (line integrals) for the specified phantom
        
    Raises
    ------
    TypeError
        If inputs are not of correct type
    ValueError
        If r and a have different shapes, are empty, or phantom_type is 
        invalid
    """
    
    # Input validation
    if not isinstance(r, np.ndarray):
        raise TypeError("r must be a numpy array")
    if not isinstance(a, np.ndarray):
        raise TypeError("a must be a numpy array")
    
    if r.shape != a.shape:
        raise ValueError("r and a must have the same shape")
    if r.size == 0:
        raise ValueError("r and a cannot be empty arrays")
    
    if not isinstance(phantom_type, str):
        raise TypeError("phantom_type must be a string")

    if phantom_type == 'bulls_eye':
        Radon = (2 * np.sqrt(np.maximum(9/16 - r**2, 0)) -
                 2 * (3/4) * np.sqrt(np.maximum(1/4 - r**2, 0)) +
                 2 * (1/4) * np.sqrt(np.maximum(1/16 - r**2, 0)))

    elif phantom_type == 'crescent_shaped':
        Radon = (2 * np.sqrt(np.maximum((1/4) - r**2, 0)) -
                 (1/2) * 2 * np.sqrt(np.maximum(
                     (9/64) - (r - (1/8) * np.cos(a))**2, 0)))

    elif phantom_type == 'shepp_logan':
        # Parameters for ellipses
        x_centers = SHEPP_LOGAN_PARAMS['x_centers']
        y_centers = SHEPP_LOGAN_PARAMS['y_centers']
        x_widths = SHEPP_LOGAN_PARAMS['x_widths']
        y_widths = SHEPP_LOGAN_PARAMS['y_widths']
        rotations = SHEPP_LOGAN_PARAMS['rotations']
        vals = SHEPP_LOGAN_PARAMS['vals']

        # Init Radon data
        Radon = np.zeros_like(r)

        for i in range(len(x_centers)):
            denominator = (y_widths[i]**2 * np.sin(a - rotations[i])**2 +
                          x_widths[i]**2 * np.cos(a - rotations[i])**2)
            numerator = (y_widths[i]**2 * np.sin(a - rotations[i])**2 +
                        x_widths[i]**2 * np.cos(a - rotations[i])**2 -
                        (r - x_centers[i] * np.cos(a) - 
                         y_centers[i] * np.sin(a))**2)
            
            Radon = Radon + (vals[i] * 2 * x_widths[i] * y_widths[i] *
                           np.sqrt(np.maximum(numerator, 0)) / denominator)

    elif phantom_type == 'smooth_phantom':
        # smoothness parameter
        alpha = SMOOTH_PHANTOM_PARAMS['alpha']

        # Parameters for ellipses
        x_centers = SMOOTH_PHANTOM_PARAMS['x_centers']
        y_centers = SMOOTH_PHANTOM_PARAMS['y_centers']
        x_widths = SMOOTH_PHANTOM_PARAMS['x_widths']
        y_widths = SMOOTH_PHANTOM_PARAMS['y_widths']
        rotations = SMOOTH_PHANTOM_PARAMS['rotations']
        vals = SMOOTH_PHANTOM_PARAMS['vals']

        # Init Radon data
        Radon = np.zeros_like(r)
        const = np.sqrt(np.pi) * gamma(alpha + 1) / gamma(alpha + 3/2)

        for i in range(len(x_centers)):
            denominator = (y_widths[i]**2 * np.sin(a - rotations[i])**2 +
                          x_widths[i]**2 * np.cos(a - rotations[i])**2)
            numerator = (y_widths[i]**2 * np.sin(a - rotations[i])**2 +
                        x_widths[i]**2 * np.cos(a - rotations[i])**2 -
                        (r - x_centers[i] * np.cos(a) - 
                         y_centers[i] * np.sin(a))**2)
            
            Radon = Radon + (vals[i] * const * x_widths[i] * y_widths[i] *
                           np.maximum(numerator, 0)**(alpha + 1/2) /
                           denominator**(alpha + 1))

    else:
        raise ValueError(f"Unknown phantom type: {phantom_type}")
    
    return Radon