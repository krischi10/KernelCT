import numpy as np
from skimage.metrics import structural_similarity as ssim

def generateGrid(xMax, yMax, size = [256, 256]):
    """
    
    """

    # Compute offset, so that points are middle points of pixels
    xOff = xMax / size[0]
    yOff = yMax / size[1]

    # Generate grid
    X,Y = np.meshgrid(np.linspace(-xMax + xOff, xMax - xOff, size[0]),
                      np.linspace(-yMax + yOff, yMax - yOff, size[1]))

    return X, Y


def generateLineset(geometry, rMax, size, seed = 0, aLim = 0):
    """
    
    """

    if geometry == 'pbg':
        # Compute number of radial samples via optimal sampling relations
        M = int(np.floor(size / np.pi))

        # Generate parameters
        r, a = np.meshgrid(np.linspace(-rMax, rMax, 2 * M + 1),
                           np.linspace(aLim, np.pi - aLim , size, endpoint = False))

    elif geometry == 'random':
        # Random parameters via random number generator
        rng = np.random.default_rng(seed)
        r = -rMax + 2 * rMax * rng.random(size)
        a = aLim + (np.pi - 2 * aLim) * rng.random(size)

    return r, a

def reconstructionError(original, reconstruction, indicator = 'rmse'):
    """
    
    """

    if indicator == 'rmse':
        val = np.sqrt( (1/original.size) * np.sum((original - reconstruction)**2) )

    elif indicator == 'ssim':
        val = ssim(original, reconstruction,
                   data_range = reconstruction.max() - reconstruction.min())

    return val



