import numpy as np
from scipy.linalg import solve, solve_triangular
from .object import Object
from .weighted_kernels import WeightedGaussian

class KernelInterpolant:

    def __init__(self, coefficients: np.ndarray | None = None) -> None:
        """
        Initialize the KernelInterpolant with optional coefficients.
        
        Parameters
        ----------
        coefficients : np.ndarray or None, default None
            Coefficients for the kernel interpolant in terms of standard basis.
            
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


    def fit(self, obj: Object, kernel: WeightedGaussian, basis: str = 'standard',
            c_newton: np.ndarray | None = None, V_newton: np.ndarray | None = None) -> None:
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
            Coefficients of interpolant in terms of Newton basis (required for 'newton' basis)
        V_newton : np.ndarray or None, optional
            Newton basis Vandermonde matrix (required for 'newton' basis)
            
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
            raise TypeError("kernel must be an instance of WeightedGaussian")
        if not isinstance(basis, str):
            raise TypeError("basis must be a string")
        
        if len(obj.ind) == 0:
            raise ValueError("Object has no indices selected for fitting.")
        
        # Validate Newton basis requirements
        if basis == 'newton':
            if c_newton is None or V_newton is None:
                raise ValueError("c_newton and V_newton must be provided for 'newton' basis")
            if not isinstance(c_newton, np.ndarray):
                raise TypeError("c_newton must be a numpy array")
            if not isinstance(V_newton, np.ndarray):
                raise TypeError("V_newton must be a numpy array")
            if c_newton.size == 0:
                raise ValueError("c_newton cannot be empty")
            if V_newton.size == 0:
                raise ValueError("V_newton cannot be empty")
            if len(obj.ind) != V_newton.shape[1]:
                raise ValueError("V_newton must have exactly as many rows as selected indices")
            if c_newton.size != V_newton.shape[1]:
                raise ValueError("c_newton size must match V_newton columns")

        # Fit the interpolant based on the specified basis
        match basis:
            case 'standard':
                self.coefficients = solve(kernel.gram_matrix(
                        obj.r[obj.ind][:, np.newaxis],
                        obj.a[obj.ind][:, np.newaxis],
                        obj.r[obj.ind][np.newaxis, :],
                        obj.a[obj.ind][np.newaxis, :]),
                    obj.Radon[obj.ind][:, np.newaxis],
                    assume_a = 'pos')
            case 'newton':
                self.coefficients = solve_triangular(V_newton[obj.ind, :].T, c_newton)
            case _:
                raise ValueError(f"Unknown basis '{basis}' specified. Must be 'standard' or 'newton'.")


    def predict(self, obj: Object, X: np.ndarray, Y: np.ndarray, kernel: WeightedGaussian,
                prev_eval_repr: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
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
            raise TypeError("kernel must be an instance of WeightedGaussian")
        if prev_eval_repr is not None and not isinstance(prev_eval_repr, np.ndarray):
            raise TypeError("prev_eval_repr must be a numpy array or None")
        
        if self.coefficients is None:
            raise ValueError("KernelInterpolant has not been fitted yet. Call fit() first.")
        
        if X.shape != Y.shape:
            raise ValueError("X and Y must have the same shape")
        if X.size == 0 or Y.size == 0:
            raise ValueError("X and Y cannot be empty")
        
        if len(obj.ind) == 0:
            raise ValueError("Object has no indices selected")
        
        # Validate prev_eval_repr if provided
        if prev_eval_repr is not None:
            if prev_eval_repr.shape[0] != X.size:
                raise ValueError("prev_eval_repr first dimension must match X.size")
        

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

        reconstruction = eval_repr @ self.coefficients

        return np.reshape(reconstruction, X.shape), eval_repr


class KernelRegressor:

    def __init__(self, coefficients: np.ndarray | None = None) -> None:
        self.coefficients = coefficients


    def fit(self, X: np.ndarray, Y: np.ndarray, Radon: np.ndarray) -> None:
        """
        """


    def predict(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        """


def reconstructKernel(self, kernel, method, gamma = None, cNewton = None,
                          V = None, returnC = False, prev_eval_repr = None):
        """
        
        """
        
        N = len(self.ind)

        # Set gamma if necessary
        if gamma is None:
                gamma = 0

        if method == 'stdIntpol':
            if N == 0:
                self.ind = range(self.r.size)
                N = len(self.ind)
            
            # Solve standard interpolation system
            cStd = solve(kernel.GramMatrix(
                        self.r[self.ind].reshape((N, 1)),
                        self.a[self.ind].reshape((N, 1)),
                        self.r[self.ind].reshape((1, N)),
                        self.a[self.ind].reshape((1, N))) \
                        + gamma * N * np.identity(N),
                    self.Radon[self.ind].reshape((N, 1)),
                    assume_a = 'pos')
            
        elif method in ['NewtonIntpol', 'NewtonRegression']:
            if method == 'NewtonRegression':
                # Regularized least squares solution
                cNewton = lsmr(V, self.Radon, damp = np.sqrt(N * gamma))[0]

            # Transformation from Newton to standard basis 
            cStd = solve_triangular(V[self.ind, :].T, cNewton)

        elif method == 'TVRegularization':

            # Set up normal equations
            A = kernel.GramMatrix(
                self.r[self.ind].reshape((N, 1)),
                self.a[self.ind].reshape((N, 1)),
                self.r[self.ind].reshape((1, N)),
                self.a[self.ind].reshape((1, N))
            )
        
            Bx = kernel.diffXRepresenter(
                self.X.reshape((self.X.size, 1)),
                self.Y.reshape((self.Y.size, 1)),
                self.r[self.ind].reshape((1, N)),
                self.a[self.ind].reshape((1, N))
            )

            By = kernel.diffYRepresenter(
                self.X.reshape((self.X.size, 1)),
                self.Y.reshape((self.Y.size, 1)),
                self.r[self.ind].reshape((1, N)),
                self.a[self.ind].reshape((1, N))
            )

            # Solve normal equations
            cStd = solve(
                A.T@A + gamma * N * (Bx.T@Bx + By.T@By),
                A.T@self.Radon[self.ind].reshape((N, 1)),
                assume_a = 'pos'
            )


        # Construct interpolant via Riesz representers
        if evalROld is None:
            indStart = 0
        else:
            indStart = evalROld.shape[1]
        
        if indStart < N:
            indR = self.ind[indStart:N]

            update = kernel.evalRepresenter(
                self.X.reshape((self.X.size, 1)),
                self.Y.reshape((self.Y.size, 1)),
                self.r[indR].reshape((1, len(indR))),
                self.a[indR].reshape((1, len(indR)))
            )

            if evalROld is None:
                evalROld = update
            else:
                evalROld = np.hstack((evalROld, update))

        reconstruction = evalROld @ cStd

        # Check if coefficients should be returned
        if returnC:
            return np.reshape(reconstruction, self.X.shape), cStd, evalROld
        return np.reshape(reconstruction, self.X.shape)