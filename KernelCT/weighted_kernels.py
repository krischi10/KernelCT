import numpy as np


class WeightedGaussian():
    """
    A class to represent a weighted Gaussian kernel.
    """
    
    def __init__(self, type, nu_s = 1, nu_w = 1):
        """
        Initialize a kernel object.
        
        Parameters:
            type (str): The type of the kernel. Currently supports 'Gaussian' as the only option.
            nu_s (float, optional): The shape parameter for the Gaussian kernel. Default is None.
            nu_w (float, optional): The weight parameter for the Gaussian kernel. Default is None.
        """

        self.type = type

        if self.type == 'Gaussian':
            self.nu_w = nu_s
            self.nu_w = nu_w


    def eval_representers(self, X, Y, r, a):
        """
        
        """
        
        if self.type == 'Gaussian':
            eval = np.sqrt(np.pi / (self.nuS + self.nuW)) * np.exp( 
                - (self.nuS +self. nuW) * (r**2 + X**2 + Y**2) \
                + 2 * self.nuS * r * (X * np.cos(a) + Y * np.sin(a)) \
                + (self.nuS**2 / (self.nuS + self.nuW))\
                    * (-X * np.sin(a) + Y * np.cos(a))**2
            )
        
        return eval
    
    def diffXRepresenter(self, X, Y, r, a):

        diffX = - (
            2 * X * (self.nuS + self.nuW) \
            - 2 * r * self.nuS * np.cos(a) \
            - 2 * (self.nuS**2 / (self.nuS + self.nuW)) * np.sin(a) \
                * (X * np.sin(a) - Y * np.cos(a))
        ) * self.evalRepresenter(X, Y, r, a)

        return diffX
    
    def diffYRepresenter(self, X, Y, r, a):

        diffY = - (
            2 * Y * (self.nuS + self.nuW) \
            - 2 * r * self.nuS * np.sin(a) \
            - 2 * (self.nuS**2 / (self.nuS + self.nuW)) * np.cos(a) \
                * (Y * np.cos(a) - X * np.sin(a))
        ) * self.evalRepresenter(X, Y, r, a)

        return diffY


    def GramMatrix(self, r, a, rC, aC):
        """
        
        """

        if self.type == 'Gaussian':
            p = (self.nuS + self.nuW) * (r**2 + rC**2)\
                - 2 * self.nuS * r * rC * np.cos(a - aC)
            q = (self.nuS + self.nuW)**2 - (self.nuS * np.cos(a - aC))**2
    
            A = (np.pi / np.sqrt(q)) * np.exp(- self.nuW\
                                              * (2 * self.nuS + self.nuW) * p / q)

        return A


    def normRadon(self, r):
        """
        
        """

        if self.type == 'Gaussian':
            norm = (np.pi / np.sqrt(self.nuW * (2 * self.nuS + self.nuW)))\
                * np.exp(-2 * self.nuW * r**2)

        return norm
    