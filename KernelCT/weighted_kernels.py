"""
Weighted Gaussian kernel implementation for CT reconstruction.

This module provides a WeightedGaussian class that implements weighted Gaussian
kernels for use in computed tomography reconstruction methods. The class includes
methods for evaluating representer functions, computing derivatives, and calculating
Gram matrices.
"""

import numpy as np


class WeightedGaussian:
    """
    A class to represent a weighted Gaussian kernel for CT reconstruction.
    """

    def __init__(self, nu_kernel: float = 2000, nu_weight: float = 1.5) -> None:
        """
        Initialize the WeightedGaussian kernel.
        
        Parameters
        ----------
        nu_kernel : float, default 2000
            Shape parameter of kernel
        nu_weight : float, default 1.5
            Shape parameter of weight function
                      
        Raises
        ------
        TypeError
            If parameters are not numeric
        ValueError
            If parameters are not positive
        """

        if not isinstance(nu_kernel, (int, float)):
            raise TypeError("nu_kernel must be a number.")
        if not isinstance(nu_weight, (int, float)):
            raise TypeError("nu_weight must be a number.")
        if nu_kernel <= 0 or nu_weight <= 0:
            raise ValueError("nu_kernel and nu_weight must be positive.")

        self.nu_kernel = nu_kernel
        self.nu_weight = nu_weight

    def eval_representers(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        r: np.ndarray,
        a: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the weighted Gaussian representer functions at given points.
        
        Parameters
        ----------
        X : np.ndarray
            X coordinates of evaluation points (2D column vector)
        Y : np.ndarray
            Y coordinates of evaluation points (2D column vector)
        r : np.ndarray
            Radial coordinates in Radon space (2D row vector)
        a : np.ndarray
            Angular coordinates in Radon space (2D row vector)
            
        Returns
        -------
        np.ndarray
            Evaluated representer function values with shape
            compatible with input coordinate arrays
        """

        # Input validation
        self._validate_representer_inputs(X, Y, r, a)

        eval = np.sqrt(np.pi / (self.nu_kernel + self.nu_weight)) * np.exp(
            - (self.nu_kernel + self.nu_weight) * (r**2 + X**2 + Y**2)
            + 2 * self.nu_kernel * r * (X * np.cos(a) + Y * np.sin(a))
            + (self.nu_kernel**2 / (self.nu_kernel + self.nu_weight))
                * (-X * np.sin(a) + Y * np.cos(a))**2
        )
        
        return eval

    def diff_x_representer(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        r: np.ndarray,
        a: np.ndarray
    ) -> np.ndarray:
        """
        Compute the partial derivative of representer functions with respect to X.
        
        Parameters
        ----------
        X : np.ndarray
            X coordinates of evaluation points (2D column vector)
        Y : np.ndarray
            Y coordinates of evaluation points (2D column vector)
        r : np.ndarray
            Radial coordinates in Radon space (2D row vector)
        a : np.ndarray
            Angular coordinates in Radon space (2D row vector)
            
        Returns
        -------
        np.ndarray
            Partial derivative values ∂/∂X of representer functions
        """

        # Input validation
        self._validate_representer_inputs(X, Y, r, a)

        diff_x = - (
            2 * X * (self.nu_kernel + self.nu_weight)
            - 2 * r * self.nu_kernel * np.cos(a)
            - 2 * (self.nu_kernel**2 / (self.nu_kernel + self.nu_weight)) * np.sin(a)
                * (X * np.sin(a) - Y * np.cos(a))
        ) * self.eval_representers(X, Y, r, a)

        return diff_x

    def diff_y_representer(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        r: np.ndarray,
        a: np.ndarray
    ) -> np.ndarray:
        """
        Compute the partial derivative of representer functions with respect to Y.
        
        Parameters
        ----------
        X : np.ndarray
            X coordinates of evaluation points (2D column vector)
        Y : np.ndarray
            Y coordinates of evaluation points (2D column vector)
        r : np.ndarray
            Radial coordinates in Radon space (2D row vector)
        a : np.ndarray
            Angular coordinates in Radon space (2D row vector)
            
        Returns
        -------
        np.ndarray
            Partial derivative values ∂/∂Y of representer functions
        """

        # Input validation
        self._validate_representer_inputs(X, Y, r, a)

        diff_y = - (
            2 * Y * (self.nu_kernel + self.nu_weight)
            - 2 * r * self.nu_kernel * np.sin(a)
            - 2 * (self.nu_kernel**2 / (self.nu_kernel + self.nu_weight)) * np.cos(a)
                * (Y * np.cos(a) - X * np.sin(a))
        ) * self.eval_representers(X, Y, r, a)

        return diff_y

    def gram_matrix(
        self,
        r: np.ndarray,
        a: np.ndarray,
        r_centers: np.ndarray,
        a_centers: np.ndarray
    ) -> np.ndarray:
        """
        Compute the Gram matrix for the weighted Gaussian kernel.
        
        Parameters
        ----------
        r : np.ndarray
            Radial coordinates for evaluation points (2D column vector)
        a : np.ndarray
            Angular coordinates for evaluation points (2D column vector)
        r_centers : np.ndarray
            Radial coordinates for kernel centers (2D row vector)
        a_centers : np.ndarray
            Angular coordinates for kernel centers (2D row vector)
            
        Returns
        -------
        np.ndarray
            Gram matrix with kernel inner products, where entry (i,j)
            represents the inner product between the respective Radon functionals
        """

        # Input validation
        self._validate_gram_matrix_inputs(r, a, r_centers, a_centers)

        p = ((self.nu_kernel + self.nu_weight) * (r**2 + r_centers**2)
             - 2 * self.nu_kernel * r * r_centers * np.cos(a - a_centers))
        q = ((self.nu_kernel + self.nu_weight)**2 
             - (self.nu_kernel * np.cos(a - a_centers))**2)

        A = ((np.pi / np.sqrt(q)) * np.exp(- self.nu_weight
                                           * (2 * self.nu_kernel + self.nu_weight) * p / q))

        return A

    def norm_radon_functional(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the norm of the Radon functional for the weighted Gaussian kernel.
        
        Parameters
        ----------
        r : np.ndarray or float
            Radial coordinates in Radon space
            
        Returns
        -------
        np.ndarray
            Norm values of the Radon functional at given radial coordinates
            
        Raises
        ------
        TypeError
            If r is not a numpy array
        """

        if not isinstance(r, (np.ndarray, float)):
            raise TypeError("r must be a numpy array or float.")

        norm = ((np.pi / np.sqrt(self.nu_weight * (2 * self.nu_kernel + self.nu_weight)))
                * np.exp(-2 * self.nu_weight * r**2))

        return norm
    
    def _validate_representer_inputs(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        r: np.ndarray,
        a: np.ndarray
    ) -> None:
        """
        Validate inputs for representer function methods.
        
        Parameters
        ----------
        X : np.ndarray
            X coordinates (should be 2D column vector)
        Y : np.ndarray
            Y coordinates (should be 2D column vector)
        r : np.ndarray
            Radial coordinates (should be 2D row vector)
        a : np.ndarray
            Angular coordinates (should be 2D row vector)
            
        Raises
        ------
        TypeError
            If inputs are not numpy arrays
        ValueError
            If inputs have incorrect shapes or dimensions
        """

        if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise TypeError("X and Y must be numpy arrays.")
        if not isinstance(r, np.ndarray) or not isinstance(a, np.ndarray):
            raise TypeError("r and a must be numpy arrays.")
        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape.")
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2D arrays.")
        if X.shape[0] == 0 or Y.shape[0] == 0:
            raise ValueError("X and Y must have at least one sample.")
        if X.shape[1] != 1 or Y.shape[1] != 1:
            raise ValueError("X and Y must be column vectors.")
        if r.shape != a.shape:
            raise ValueError("r and a must have the same shape.")
        if r.ndim != 2 or a.ndim != 2:
            raise ValueError("r and a must be 2D arrays.")
        if r.shape[1] == 0 or a.shape[1] == 0:
            raise ValueError("r and a must have at least one sample.")
        if r.shape[0] != 1 or a.shape[0] != 1:
            raise ValueError("r and a must be row vectors.")

    def _validate_gram_matrix_inputs(
        self,
        r: np.ndarray,
        a: np.ndarray,
        r_centers: np.ndarray,
        a_centers: np.ndarray
    ) -> None:
        """
        Validate inputs for Gram matrix computation.
        
        Parameters
        ----------
        r : np.ndarray
            Radial coordinates for evaluation points (should be 2D column vector)
        a : np.ndarray
            Angular coordinates for evaluation points (should be 2D column vector)
        r_centers : np.ndarray
            Radial coordinates for kernel centers (should be 2D row vector)
        a_centers : np.ndarray
            Angular coordinates for kernel centers (should be 2D row vector)
            
        Raises
        ------
        TypeError
            If inputs are not numpy arrays
        ValueError
            If inputs have incorrect shapes or dimensions
        """

        if not isinstance(r, np.ndarray) or not isinstance(a, np.ndarray):
            raise TypeError("r and a must be numpy arrays.")
        if not isinstance(r_centers, np.ndarray) or not isinstance(a_centers, np.ndarray):
            raise TypeError("r_centers and a_centers must be numpy arrays.")
        if r.shape != a.shape:
            raise ValueError("r and a must have the same shape.")
        if r.ndim != 2 or a.ndim != 2:
            raise ValueError("r and a must be 2D arrays.")
        if r.shape[1] != 1 or a.shape[1] != 1:
            raise ValueError("r and a must be column vectors.")
        if r.shape[0] == 0 or a.shape[0] == 0:
            raise ValueError("r and a must have at least one sample.")
        if r_centers.shape != a_centers.shape:
            raise ValueError("r_centers and a_centers must have the same shape.")
        if r_centers.ndim != 2 or a_centers.ndim != 2:
            raise ValueError("r_centers and a_centers must be 2D arrays.")
        if r_centers.shape[0] != 1 or a_centers.shape[0] != 1:
            raise ValueError("r_centers and a_centers must be row vectors.")
        if r.shape[1] == 0 or a.shape[1] == 0:
            raise ValueError("r and a must have at least one sample.")