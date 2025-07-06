import numpy as np
from scipy.linalg import solve, solve_triangular
from scipy.sparse.linalg import lsmr
from .object import Object
from .weighted_kernels import WeightedGaussian


class KernelReconstructor:
    """
    Base class for kernel-based reconstruction methods.
    
    Provides common functionality for kernel-based reconstruction including
    coefficient storage and prediction methods.
    """
    def __init__(self, coefficients: np.ndarray | None = None) -> None:
        """
        Initialize the KernelReconstructor with optional coefficients.
        
        Parameters
        ----------
        coefficients : np.ndarray or None, default None
            Coefficients for the kernel reconstructor.
            
        Raises
        ------
        TypeError
            If coefficients is not None and not a numpy array
        ValueError
            If coefficients is empty
        """
        
        # Input validation
        if coefficients is not None:
            if not isinstance(coefficients, np.ndarray):
                raise TypeError("coefficients must be a numpy array or None")
            if coefficients.size == 0:
                raise ValueError("coefficients cannot be empty")
        
        self.coefficients = coefficients


    def predict(
        self, obj: Object, X: np.ndarray, Y: np.ndarray, 
        kernel: WeightedGaussian,
        prev_eval_repr: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict using the fitted kernel interpolant.
        
        Parameters
        ----------
        obj : Object
            Object containing the data and selected indices
        X : np.ndarray
            X coordinates for prediction grid
        Y : np.ndarray
            Y coordinates for prediction grid
        kernel : WeightedGaussian
            Kernel instance for evaluation
        prev_eval_repr : np.ndarray or None, optional
            Previously computed representer evaluations for incremental computation

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Reconstruction with same shape as X and Y, and representer evaluations
            
        Raises
        ------
        TypeError
            If inputs are not of correct type
        ValueError
            If interpolant has not been fitted, arrays have incompatible shapes,
            or prev_eval_repr has invalid dimensions
        """
        
        # Input validation
        if not isinstance(obj, Object):
            raise TypeError("obj must be an instance of Object")
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if not isinstance(Y, np.ndarray):
            raise TypeError("Y must be a numpy array")
        if not isinstance(kernel, WeightedGaussian):
            raise TypeError(
                "kernel must be an instance of WeightedGaussian"
            )
        if (prev_eval_repr is not None and 
            not isinstance(prev_eval_repr, np.ndarray)):
            raise TypeError(
                "prev_eval_repr must be a numpy array or None"
            )
        
        if self.coefficients is None:
            raise ValueError(
                "KernelReconstructor has not been fitted yet. "
                "Call fit() first."
            )
        
        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape")
        if X.size == 0 or Y.size == 0:
            raise ValueError("X and Y cannot be empty")
        
        if len(obj.ind) == 0:
            raise ValueError("Object has no indices selected")
        
        # Validate prev_eval_repr if provided
        if prev_eval_repr is not None:
            if prev_eval_repr.shape[0] != X.size:
                raise ValueError(
                    "prev_eval_repr first dimension must match X.size"
                )
    

        if prev_eval_repr is None:
            start_index = 0
        else:
            start_index = prev_eval_repr.shape[1]

        if start_index < self.coefficients.size:
            update_range_repr = obj.ind[start_index:self.coefficients.size]

            update = kernel.eval_representers(
                X.reshape((X.size, 1)),
                Y.reshape((Y.size, 1)),
                obj.r[update_range_repr][np.newaxis, :],
                obj.a[update_range_repr][np.newaxis, :]
            )

            if prev_eval_repr is None:
                eval_repr = update
            else:
                eval_repr = np.hstack((prev_eval_repr, update))
        else:
            eval_repr = prev_eval_repr

        reconstruction = eval_repr @ self.coefficients

        return np.reshape(reconstruction, X.shape), eval_repr


class KernelInterpolant(KernelReconstructor):
    """
    Kernel interpolation for CT reconstruction.
    
    Performs exact interpolation of selected Radon transform data using
    kernel methods with standard or Newton basis.
    """
    def fit(
        self, obj: Object, kernel: WeightedGaussian, 
        basis: str = 'standard',
        c_newton: np.ndarray | None = None, 
        V_newton: np.ndarray | None = None
    ) -> None:
        """
        Fit the kernel interpolant to the selected data points.
        
        Parameters
        ----------
        obj : Object
            Object containing the data and selected indices
        kernel : WeightedGaussian
            Kernel instance for interpolation
        basis : str, default 'standard'
            Basis type for fitting ('standard' or 'newton')
        c_newton : np.ndarray or None, optional
            Coefficients of interpolant in terms of Newton basis 
            (required for 'newton' basis)
        V_newton : np.ndarray or None, optional
            Newton basis Vandermonde matrix 
            (required for 'newton' basis)
            
        Raises
        ------
        TypeError
            If inputs are not of correct type
        ValueError
            If Object has no indices selected, basis is invalid, or required
            parameters are missing for Newton basis
        """
        
        # Input validation
        if not isinstance(obj, Object):
            raise TypeError("obj must be an instance of Object")
        if not isinstance(kernel, WeightedGaussian):
            raise TypeError(
                "kernel must be an instance of WeightedGaussian"
            )
        if not isinstance(basis, str):
            raise TypeError("basis must be a string")
    
        if len(obj.ind) == 0:
            raise ValueError("Object has no indices selected for fitting.")
        
        # Validate Newton basis requirements
        if basis == 'newton':
            if c_newton is None or V_newton is None:
                raise ValueError(
                    "c_newton and V_newton must be provided for "
                    "'newton' basis"
                )
            if not isinstance(c_newton, np.ndarray):
                raise TypeError("c_newton must be a numpy array")
            if not isinstance(V_newton, np.ndarray):
                raise TypeError("V_newton must be a numpy array")
            if c_newton.size == 0:
                raise ValueError("c_newton cannot be empty")
            if V_newton.size == 0:
                raise ValueError("V_newton cannot be empty")
            if len(obj.ind) != V_newton.shape[1]:
                raise ValueError(
                    "V_newton must have exactly as many rows as "
                    "selected indices"
                )
            if c_newton.size != V_newton.shape[1]:
                raise ValueError(
                    "c_newton size must match V_newton columns"
                )

        # Fit the interpolant based on the specified basis
        match basis:
            case 'standard':
                self.coefficients = solve(
                    kernel.gram_matrix(
                        obj.r[obj.ind][:, np.newaxis],
                        obj.a[obj.ind][:, np.newaxis],
                        obj.r[obj.ind][np.newaxis, :],
                        obj.a[obj.ind][np.newaxis, :]
                    ),
                    obj.Radon[obj.ind][:, np.newaxis],
                    assume_a='pos'
                )
            case 'newton':
                self.coefficients = solve_triangular(
                    V_newton[obj.ind, :].T, c_newton
                )
            case _:
                raise ValueError(
                    f"Unknown basis '{basis}' specified. "
                    "Must be 'standard' or 'newton'."
                )


class KernelRegressor(KernelReconstructor):
    """
    Kernel regression for CT reconstruction with regularization.
    
    Performs regularized reconstruction of Radon transform data using
    total variation or norm-based regularization methods.
    """
    def fit(
        self, obj: Object, kernel: WeightedGaussian, 
        regularization: str, gamma: float = 0.0, 
        X_diff: np.ndarray | None = None, 
        Y_diff: np.ndarray | None = None, 
        V_newton: np.ndarray | None = None
    ) -> None:
        """
        Fit the kernel regressor with regularization to the Radon transform data.
        
        Parameters
        ----------
        obj : Object
            Object containing the Radon transform data and selected indices
        kernel : WeightedGaussian
            Kernel instance for regression
        regularization : str
            Regularization method ('tv' for total variation, 
            'norm' for norm-based)
        gamma : float, default 0.0
            Regularization parameter
        X_diff : np.ndarray or None, optional
            X coordinates for gradient evaluation 
            (required for 'tv' regularization)
        Y_diff : np.ndarray or None, optional
            Y coordinates for gradient evaluation 
            (required for 'tv' regularization)
        V_newton : np.ndarray or None, optional
            Newton basis matrix (required for 'norm' regularization)
            
        Raises
        ------
        ValueError
            If unknown regularization method is specified or required 
            parameters are missing for the chosen regularization method
        """

        # Input validation
        if not isinstance(obj, Object):
            raise TypeError("obj must be an instance of Object")
        if not isinstance(kernel, WeightedGaussian):
            raise TypeError(
                "kernel must be an instance of WeightedGaussian"
            )
        if not isinstance(regularization, str):
            raise TypeError("regularization must be a string")
        if not isinstance(gamma, (int, float)):
            raise TypeError("gamma must be a number")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        
        if len(obj.ind) == 0:
            raise ValueError(
                "Object has no indices selected for fitting"
            )
        
        # Validate TV regularization requirements
        if regularization == 'tv':
            if X_diff is None or Y_diff is None:
                raise ValueError(
                    "X_diff and Y_diff must be provided for "
                    "'tv' regularization"
                )
            if not isinstance(X_diff, np.ndarray):
                raise TypeError("X_diff must be a numpy array")
            if not isinstance(Y_diff, np.ndarray):
                raise TypeError("Y_diff must be a numpy array")
            if X_diff.shape != Y_diff.shape:
                raise ValueError(
                    "X_diff and Y_diff must have the same shape"
                )
            if X_diff.size == 0 or Y_diff.size == 0:
                raise ValueError(
                    "X_diff and Y_diff cannot be empty"
                )
        
        # Validate norm regularization requirements
        if regularization == 'norm':
            if V_newton is None:
                raise ValueError(
                    "V_newton must be provided for 'norm' regularization"
                )
            if not isinstance(V_newton, np.ndarray):
                raise TypeError("V_newton must be a numpy array")
            if V_newton.size == 0:
                raise ValueError("V_newton cannot be empty")
            if V_newton.shape[1] != len(obj.ind):
                raise ValueError(
                    "V_newton must have exactly as many columns as "
                    "selected indices"
                )

        match regularization:
            case 'tv':
                # Set up normal equations
                A = kernel.gram_matrix(
                    obj.r[:, np.newaxis],
                    obj.a[:, np.newaxis],
                    obj.r[obj.ind][np.newaxis, :],
                    obj.a[obj.ind][np.newaxis, :]
                )
            
                diff_x = kernel.diff_x_representer(
                    X_diff.reshape((X_diff.size, 1)),
                    Y_diff.reshape((Y_diff.size, 1)),
                    obj.r[obj.ind][np.newaxis, :],
                    obj.a[obj.ind][np.newaxis, :]
                )

                diff_y = kernel.diff_y_representer(
                    X_diff.reshape((X_diff.size, 1)),
                    Y_diff.reshape((Y_diff.size, 1)),
                    obj.r[obj.ind][np.newaxis, :],
                    obj.a[obj.ind][np.newaxis, :]
                )

                # Solve normal equations
                self.coefficients = solve(
                    A.T@A + gamma * obj.Radon.size * (
                        diff_x.T@diff_x + diff_y.T@diff_y
                    ),
                    A.T@obj.Radon[:, np.newaxis],
                    assume_a='pos'
                )

            case 'norm':
                c_reg_newton = lsmr(
                    V_newton, obj.Radon, 
                    damp=np.sqrt(obj.Radon.size * gamma)
                )[0]
                self.coefficients = solve_triangular(
                    V_newton[obj.ind, :].T, c_reg_newton
                )

            case _:
                raise ValueError(
                    f"Unknown regularization method '{regularization}'. "
                    "Must be 'tv' or 'norm'."
                )