import numpy as np



def evalDist(greedyMethod, kernel, r, a, rC, aC, s = None):
    """
    
    """

    if greedyMethod == 'geoDual':
        dist = kernel.normRadon(r) - 2 * kernel.GramMatrix(r, a, rC, aC) \
            + kernel.normRadon(rC)
        
    elif greedyMethod == 'geoParam':
        dist = (r - rC)**2 + (a - aC)**2

    elif greedyMethod == 'geoPeriodic':
        dist = np.min([
            (r - rC)**2 + (a - aC)**2,
            (r + rC)**2 + (a - aC + np.pi)**2,
            (r + rC)**2 + (a - aC - np.pi)**2
        ], axis = 0)
        
    elif greedyMethod == 'geoSphere':
        innerProd = np.clip(
            (1 / np.sqrt(1 + r**2 * s**2)) * (1 / np.sqrt(1 + rC**2 * s**2)) \
                * (np.cos(a - aC) + r * rC * s**2),
            -1,
            1
        )           # avoid leaving the domain of arccos via truncation
        dist = np.minimum(np.arccos(innerProd), np.arccos(-innerProd))

    return dist


def betaGreedy(dist, res, beta):
    """
    
    """

    indices = list(np.where(dist >= 1e-20)[0])  # avoid division by zero 

    if beta == 'inf':
        i = np.argmax(np.absolute(res[indices]) / np.sqrt(dist[indices]))
    else:
        i = np.argmax(np.absolute(res[indices])**beta * np.sqrt(dist[indices])**(1-beta))

    return indices[i]
