"""
Object class for CT reconstruction using kernel methods.

This module provides the Object class that handles Radon transform data
and implements the filtered back projection method and kernel-based thinning
via greedy algorithms.
"""

import numpy as np
from . import fbp
from .weighted_kernels import WeightedGaussian
from . import greedy_methods as greedy
import tqdm

class Object:
    """
    An object class for fbp and kernel-based CT reconstruction.
    """

    def __init__(
        self,
        r: np.ndarray,
        a: np.ndarray,
        Radon: np.ndarray,
        ind: list[int] = []
    ) -> None:
        """
        Initialize the Object class with Radon transform parameters.
        
        Parameters
        ----------
        r : np.ndarray
            Radial coordinates for Radon transform
        a : np.ndarray
            Angular coordinates for Radon transform
        Radon : np.ndarray
            Radon transform data values
        ind : list[int], default []
            List of selected indices for kernel methods
            
        Raises
        ------
        TypeError
            If inputs are not of correct type or ind contains non-integers
        ValueError
            If r and a have different shapes or any arrays are empty
        """
        
        # Input validation
        if not isinstance(r, np.ndarray):
            raise TypeError("r must be a numpy array")
        if not isinstance(a, np.ndarray):
            raise TypeError("a must be a numpy array")
        if not isinstance(Radon, np.ndarray):
            raise TypeError("Radon must be a numpy array")
        if not isinstance(ind, list):
            raise TypeError("ind must be a list of indices")
        # check if ind is list of integers
        if not all(isinstance(i, int) for i in ind):
            raise TypeError("ind must be a list of integers")

        if r.shape != a.shape:
            raise ValueError("r and a must have the same shape")
        if r.size == 0 or a.size == 0 or Radon.size == 0:
            raise ValueError("r, a, and Radon cannot be empty arrays")

        # Assign parameters
        self.r = r
        self.a = a
        self.Radon = Radon
        self.ind = ind

    def fbp_reconstruct(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        filter_type: str,
        spline_type: str,
        eval_radius: float
    ) -> np.ndarray:
        """
        Perform filtered back projection reconstruction using the Radon transform data.
        
        Parameters
        ----------
        X : np.ndarray
            X coordinates for reconstruction grid
        Y : np.ndarray
            Y coordinates for reconstruction grid
        filter_type : str
            Type of filter to use ('ram_lak' or 'shepp_logan')
        spline_type : str
            Type of spline interpolation ('linear' or 'cubic')
        eval_radius : float
            Evaluation radius for reconstruction, make sure that any
            (X, Y) coordinates are within this radius
            
        Returns
        -------
        np.ndarray
            Reconstructed image with same shape as X and Y
            
        Raises
        ------
        ValueError
            If arrays are empty, X and Y have different shapes, eval_radius is not positive,
            filter_type or spline_type are invalid, or r and a are not 2D arrays
        """

        # Input validation
        if self.r.size == 0 or self.a.size == 0 or self.Radon.size == 0:
            raise ValueError("Some of the input arrays are empty.")
        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape.")
        if eval_radius <= 0:
            raise ValueError("eval_radius must be positive.")

        # check filter type
        if filter_type not in ['ram_lak', 'shepp_logan']:
            raise ValueError("filter_type must be one of: 'ram_lak', 'shepp_logan'")

        # check spline type
        if spline_type not in ['linear', 'cubic']:
            raise ValueError("spline_type must be one of: 'linear', 'cubic'")

        # verify that r and a are 2D arrays
        if self.r.ndim != 2 or self.a.ndim != 2:
            raise ValueError("r and a must be 2D arrays")

        # compute parameters via optimal sampling equalities
        r_max = np.max(np.abs(self.r))
        L = self.a.shape[0] / r_max
        d = np.pi / L
        filter_width = int((self.a.shape[1] - 1) / 2) + int(np.ceil(eval_radius / d))
        spline_data_length = int(np.ceil(eval_radius / d))

        # setup data for spline interpolation
        x_eval = (np.cos(self.a[:, [0]]) * X.reshape((1, X.size))
                  + np.sin(self.a[:, [0]]) * Y.reshape((1, Y.size)))
        x_data = d * np.arange(-spline_data_length, spline_data_length + 1)

        # compute low pass filtering
        filtered_data = d * fbp.low_pass_filter(
            self.Radon,
            filter_type,
            L,
            filter_width
        )
        
        # spline interpolation + back projection
        reconstruction = fbp.back_projection(
            fbp.row_wise_splines(x_eval, x_data, filtered_data, spline_type)
        )

        return np.reshape(reconstruction, X.shape)

    def kernel_thinning(
        self,
        kernel: WeightedGaussian,
        greedy_method: str,
        max_iter: int,
        tol: float = 1e-15,
        beta: float | int | str = None,
        c_newton: np.ndarray = None,
        V_newton: np.ndarray = None,
        pwr_func_vals: np.ndarray = None,
        res: np.ndarray = None,
        dist: np.ndarray = None,
        s: float = 1.2
    ) -> tuple:
        """
        Perform kernel thinning using greedy algorithms for optimal sampling.
        
        Parameters
        ----------
        kernel : WeightedGaussian
            Kernel instance for thinning and later reconstruction
        greedy_method : str
            Greedy selection method ('beta_greedy', 'geo_dual_space', 
            'geo_parameter_space', 'geo_periodic', 'geo_sphere')
        max_iter : int
            Maximum number of iterations
        tol : float, default 1e-15
            Tolerance for power function convergence
        beta : float or int or str, optional
            Beta parameter for beta_greedy method
        c_newton : np.ndarray, optional
            Newton coefficients for continuation
        V_newton : np.ndarray, optional
            Newton basis matrix for continuation
        pwr_func_vals : np.ndarray, optional
            Power function values for continuation
        res : np.ndarray, optional
            Residual values for continuation
        dist : np.ndarray, optional
            Distance array for continuation of geometric methods
        s : float, default 1.2
            Scaling parameter for geo_sphere method
            
        Returns
        -------
        tuple
            For beta_greedy: (c_newton, V_newton, pwr_func_vals, res)
            For geometric methods: (c_newton, V_newton, pwr_func_vals, res, dist)
            
        Raises
        ------
        TypeError
            If inputs are not of correct type
        ValueError
            If parameters are invalid or required parameters are missing
        """

        # Input validation
        self._validate_kernel_thinning_inputs(
            kernel=kernel,
            greedy_method=greedy_method,
            max_iter=max_iter,
            tol=tol,
            beta=beta,
            c_newton=c_newton,
            V_newton=V_newton,
            pwr_func_vals=pwr_func_vals,
            res=res,
            dist=dist,
            s=s
        )

        # Initialize thinning procedure
        if len(self.ind) == 0:
            # start with the first functional
            self.ind.append(0)
            not_in_ind = list(range(1, self.r.size))
            start = 1
            end = min(max_iter, self.r.size - 1)

            V_newton = np.zeros((self.r.size, max_iter))
            V_newton[:, [0]] = (
                (1 / np.sqrt(kernel.norm_radon_functional(self.r[0])))
                * kernel.gram_matrix(
                    self.r[:, np.newaxis], self.a[:, np.newaxis],
                    np.array([[self.r[0]]]), np.array([[self.a[0]]])
                )
            )

            pwr_func_vals = (
                kernel.norm_radon_functional(self.r) - V_newton[:, 0]**2
            )
            c_newton = np.zeros((max_iter,))
            c_newton[0] = (
                self.Radon[0] / np.sqrt(kernel.norm_radon_functional(self.r[0]))
            )
            res = self.Radon - c_newton[0] * V_newton[:, 0]

            if greedy_method in [
                'geo_dual_space', 'geo_parameter_space', 'geo_periodic', 'geo_sphere'
            ]:
                dist = np.inf * np.ones_like(self.r)
        else:
            not_in_ind = [
                index for index in range(self.r.size) if index not in self.ind
            ]
            start = len(self.ind)
            end = len(self.ind) + min(max_iter, self.r.size - len(self.ind))
            V_newton = np.hstack(
                (V_newton, np.zeros((self.r.size, max_iter)))
            )
            c_newton = np.hstack((c_newton, np.zeros((max_iter,))))

        # Greedy selection loop
        for k in tqdm.tqdm(
            range(start, end), desc=f'{greedy_method}',
            bar_format='{l_bar}{bar:30}{r_bar}',
            total=end, initial=start
        ):

            # Greedy selection
            match greedy_method:
                case 'beta_greedy':
                    i = greedy.beta_greedy(
                        pwr_func_vals[not_in_ind], res[not_in_ind], beta
                    )
                case 'geo_dual_space' | 'geo_parameter_space' | 'geo_periodic' | 'geo_sphere':
                    dist[not_in_ind], i = greedy.geometric_greedy(
                        dist[not_in_ind], greedy_method, self.r[not_in_ind],
                        self.a[not_in_ind], self.r[index], self.a[index],
                        kernel=kernel, s=s
                    )
                case _:
                    raise ValueError(
                        "greedy_method must be one of: 'beta_greedy', "
                        "'geo_dual_space', 'geo_parameter_space', 'geo_periodic', "
                        "'geo_sphere'."
                    )
            index = not_in_ind[i]

            # Determine power function value at chosen functional
            # Break if power function too small -> numerical instability
            p = np.sqrt(pwr_func_vals[index])

            if p < tol:
                c_newton = c_newton[:k]
                V_newton = V_newton[:, :k]
                print('Power function too small, stopping greedy selection.')
                break

            # Update Newton basis
            V_newton[not_in_ind, [k]] = np.ravel(
                (1/p) * (
                    kernel.gram_matrix(
                        self.r[not_in_ind][:, np.newaxis],
                        self.a[not_in_ind][:, np.newaxis],
                        np.array([[self.r[index]]]),
                        np.array([[self.a[index]]])
                    ) - (V_newton[not_in_ind, :k] @ V_newton[[index], :k].T)
                )
            )
            
            # Update power function values
            pwr_func_vals[not_in_ind] = (
                pwr_func_vals[not_in_ind] - V_newton[not_in_ind, k]**2
            )

            # Determine coefficient of interpolant and update residual
            c_newton[k] = res[index] / p
            res[not_in_ind] = (
                res[not_in_ind] - c_newton[k] * V_newton[not_in_ind, k]
            )

            # Modification of index sets
            self.ind.append(index)
            not_in_ind.remove(index)
        # end loop

        # conditional return
        match greedy_method:
            case 'beta_greedy':
                return c_newton, V_newton, pwr_func_vals, res
            case 'geo_dual_space' | 'geo_parameter_space' | 'geo_periodic' | 'geo_sphere':
                return c_newton, V_newton, pwr_func_vals, res, dist

    def _validate_kernel_thinning_inputs(
        self,
        kernel: WeightedGaussian,
        greedy_method: str,
        max_iter: int,
        tol: float,
        beta: float | int | str,
        c_newton: np.ndarray,
        V_newton: np.ndarray,
        pwr_func_vals: np.ndarray,
        res: np.ndarray,
        dist: np.ndarray,
        s: float
    ) -> None:
        """
        Validate inputs for the kernel thinning method.
        
        Parameters
        ----------
        kernel : WeightedGaussian
            Kernel instance for reconstruction
        greedy_method : str
            Greedy selection method name
        max_iter : int
            Maximum number of iterations
        tol : float, default 1e-15
            Tolerance for convergence
        beta : float or int or str, optional
            Beta parameter for beta_greedy method
        c_newton : np.ndarray, optional
            Newton coefficients
        V_newton : np.ndarray, optional
            Newton basis matrix
        pwr_func_vals : np.ndarray, optional
            Power function values
        res : np.ndarray, optional
            Residual values
        dist : np.ndarray, optional
            Distance array for geometric methods
        s : float, optional
            Scaling parameter for geo_sphere method
            
        Raises
        ------
        TypeError
            If inputs are not of correct type
        ValueError
            If r and a are not 1D arrays, parameters are invalid, or required
            parameters are missing for specific methods
        """

        # check if r, a are 1D array
        if self.r.ndim != 1 or self.a.ndim != 1:
            raise ValueError("r and a must be 1D arrays")
        if not isinstance(kernel, WeightedGaussian):
            raise TypeError("kernel must be an instance of WeightedGaussian")
        if not isinstance(greedy_method, str):
            raise TypeError("greedy_method must be a string")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if not isinstance(beta, (float, int, str)):
            raise TypeError("beta must be a float, int, or 'inf'")
        if not isinstance(tol, float) or tol <= 0:
            raise ValueError("tol must be a positive float")
        if greedy_method == 'beta_greedy' and beta is None:
            raise ValueError("beta must be provided for beta_greedy method")
        if len(self.ind) > 0:
            if (c_newton is None or V_newton is None or
                pwr_func_vals is None or res is None):
                raise ValueError(
                    "c_newton, V_newton, pwr_func_vals, and res must be "
                    "provided for non-empty ind"
                )
            if (not isinstance(c_newton, np.ndarray) or
                not isinstance(V_newton, np.ndarray)):
                raise TypeError("c_newton and V_newton must be numpy arrays")
            if (not isinstance(pwr_func_vals, np.ndarray) or
                not isinstance(res, np.ndarray)):
                raise TypeError("pwr_func_vals and res must be numpy arrays")
        if greedy_method in [
            'geo_dual_space', 'geo_parameter_space', 'geo_periodic', 'geo_sphere'
        ]:
            if len(self.ind) > 0 and dist is None:
                raise ValueError(
                    "dist must be provided for geometric greedy methods"
                )
        if greedy_method == 'geo_sphere':
            if not isinstance(s, (float, int)):
                raise TypeError("s must be a float or int")
            if s <= 0:
                raise ValueError(
                    "Scale parameter s must be positive for geo_sphere method"
                )