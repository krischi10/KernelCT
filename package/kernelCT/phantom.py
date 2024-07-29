import numpy as np
from scipy.special import gamma


def getPhantomEval(type, X, Y):
    """
    
    """

    if type == 'bullseye':
        eval = (X**2 + Y**2 <= 9/16) - (3/4) * (X**2 + Y**2 <= 1/4) \
                + (1/4) * (X**2 + Y**2 <= 1/16)

    elif type == 'crescentShaped':
        eval = (X**2 + Y**2 <= 1/4) - (1/2) * (((X - (1/8))**2 + Y**2) <= (9/64))

    elif type == 'SheppLogan':
        # Parameters for ellipses
        xC = np.array([0, 0, 0.22, -0.22, 0, 0, 0, -0.08, 0, 0.06])
        yC = np.array([0, -0.0184, 0, 0, 0.35, 0.1, -0.1, -0.605, -0.605, -0.605])
        widthX = np.array([0.69, 0.6624, 0.11, 0.16, 0.21,
                           0.046, 0.046, 0.046, 0.023, 0.023])
        widthY = np.array([0.92, 0.874, 0.31, 0.41, 0.25,
                           0.046, 0.046, 0.023, 0.023, 0.046])
        theta = np.array([0, 0, -np.pi/10, np.pi/10, 0, 0, 0, 0, 0, 0])
        f = np.array([1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        # Initialize point eval loop over ellipses
        eval = np.zeros_like(X)

        for i in range(len(xC)):

            eval = eval + f[i] * ( 
                    ((X - xC[i])*np.cos([theta[i]]) + (Y - yC[i])*np.sin(theta[i]))**2\
                        / widthX[i]**2 \
                    + ( -(X - xC[i])*np.sin(theta[i]) + (Y - yC[i])*np.cos(theta[i]))**2\
                        / widthY[i]**2 <=1
                )
    
    elif type == 'smoothPhantom':
        # smoothness parameter
        alpha = 3

        # Parameters for ellipses
        xC = np.array([0.22, -0.22, 0])
        yC = np.array([0, 0, 0.2])
        widthX = np.array([0.51, 0.51, 0.5])
        widthY = np.array([0.31, 0.36, 0.8])
        theta = np.array([(2/5)*np.pi, (3/5)*np.pi, np.pi/2])
        f = np.array([1, -3/2, 3/2])

        # Initialize point eval and Radon data + loop over ellipses
        eval = np.zeros_like(X)

        for i in range(len(xC)):

            eval = eval + f[i] * np.maximum(
                1 - ((X - xC[i])*np.cos([theta[i]]) \
                    + (Y - yC[i])*np.sin(theta[i]))**2 / widthX[i]**2 \
                    - ( -(X - xC[i])*np.sin(theta[i]) \
                    + (Y - yC[i])*np.cos(theta[i]) )**2 / widthY[i]**2,
                0)**alpha

    return eval

def getPhantomRadon(type, r, a):
    
    if type == 'bullseye':
        Radon = 2 * np.sqrt(np.maximum(9/16 - r**2, 0)) \
                - 2 * (3/4) * np.sqrt(np.maximum(1/4 - r**2, 0)) \
                + 2 * (1/4) * np.sqrt(np.maximum(1/16 - r**2, 0))
    
    elif type == 'crescentShaped':
        Radon = 2 * np.sqrt(np.maximum((1/4) - r**2, 0)) \
                - (1/2)*2 * np.sqrt(np.maximum((9/64) - (r - (1/8)*np.cos(a))**2, 0))
        
    elif type == 'SheppLogan':
        # Parameters for ellipses
        xC = np.array([0, 0, 0.22, -0.22, 0, 0, 0, -0.08, 0, 0.06])
        yC = np.array([0, -0.0184, 0, 0, 0.35, 0.1, -0.1, -0.605, -0.605, -0.605])
        widthX = np.array([0.69, 0.6624, 0.11, 0.16, 0.21,
                           0.046, 0.046, 0.046, 0.023, 0.023])
        widthY = np.array([0.92, 0.874, 0.31, 0.41, 0.25,
                           0.046, 0.046, 0.023, 0.023, 0.046])
        theta = np.array([0, 0, -np.pi/10, np.pi/10, 0, 0, 0, 0, 0, 0])
        f = np.array([1, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        # Init Radon data
        Radon = np.zeros_like(r)

        for i in range(len(xC)):
            Radon = Radon + f[i] * 2 * widthX[i] * widthY[i]\
                * np.sqrt(np.maximum(
                        widthY[i]**2 * np.sin(a - theta[i])**2\
                            + widthX[i]**2 * np.cos(a - theta[i])**2 \
                            - (r - xC[i] * np.cos(a) - yC[i] * np.sin(a))**2,
                        0)
                    ) \
                    / (widthY[i]**2 * np.sin(a - theta[i])**2\
                       + widthX[i]**2 * np.cos(a - theta[i])**2)
    
    elif type == 'smoothPhantom':
        # smoothness parameter
        alpha = 3

        # Parameters for ellipses
        xC = np.array([0.22, -0.22, 0])
        yC = np.array([0, 0, 0.2])
        widthX = np.array([0.51, 0.51, 0.5])
        widthY = np.array([0.31, 0.36, 0.8])
        theta = np.array([(2/5)*np.pi, (3/5)*np.pi, np.pi/2])
        f = np.array([1, -3/2, 3/2])

        # Init Radon data
        Radon = np.zeros_like(r)
        const = np.sqrt(np.pi) * gamma(alpha + 1) / gamma(alpha + 3/2)

        for i in range(len(xC)):
            Radon = Radon + f[i] * const * widthX[i] * widthY[i]\
                * np.maximum(
                        widthY[i]**2 * np.sin(a - theta[i])**2 \
                        + widthX[i]**2 * np.cos(a - theta[i])**2 \
                        - (r - xC[i] * np.cos(a) - yC[i] * np.sin(a))**2,
                    0)**(alpha + 1/2) \
                    / (widthY[i]**2 * np.sin(a - theta[i])**2\
                       + widthX[i]**2 * np.cos(a - theta[i])**2)**(alpha + 1)
    
    return Radon
