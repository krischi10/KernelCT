import numpy as np
from scipy.interpolate import CubicSpline


def FourierLP(filter, fWidth, width):
    """
    
    """

    # Indices for radial variable discretization
    t = np.arange(-width, width + 1)

    if filter == 'RamLak':
        fourier = (fWidth**2 / (2 * np.pi)) * (2 * np.sinc(t) - np.sinc(t/2)**2) 

    elif filter == 'SheppLogan':
        fourier = 4 * fWidth**2 / (np.pi**3 * (1 - 4 * t**2))

    return fourier


def convRows(Radon, convFilter):
    """
    
    """

    # Initialize convolution
    convolution = np.zeros((Radon.shape[0], np.size(convFilter) - Radon.shape[1] + 1))
    
    for i in range(np.shape(Radon)[0]):
        convolution[i,:] = np.convolve(convFilter, Radon[i, :], mode = 'valid')

    return convolution


def rowSplines(xEval, xData, vals, spline = 'linear'):
    """
    
    """

    splineEval = np.zeros_like(xEval)

    if spline == 'linear':
        for i in range(xEval.shape[0]):
            splineEval[i, :] = np.interp(xEval[i, :], xData, vals[i, :])

    elif spline == 'cubic':
        for i in range(xEval.shape[0]):         
            cs = CubicSpline(xData, vals[i, :])
            splineEval[i, :] = cs(xEval[i, :])

    return splineEval


def backProjection(h):
    """
    
    """

    return (1 / (2 * np.shape(h)[0])) * np.sum(h, axis = 0)