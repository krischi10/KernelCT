import numpy as np
from . import fbp
import tqdm

class Object:
    """
    
    """

    def __init__(
        self,
        r: np.ndarray,
        a: np.ndarray,
        Radon: np.ndarray,
        ind: list[int] = []
    ) -> None:
        """
        Initialize the Object class with Radon transform parameters
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
        x_eval = np.cos(self.a[:, [0]]) * X.reshape((1, X.size)) \
                + np.sin(self.a[:, [0]]) * Y.reshape((1, Y.size))
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


    # def kernelThinning(self, kernel, greedyMethod, maxIter, mode = 'reconstruct',
    #                     beta = None, tol = 1e-15, c = None, V = None, dist = None,
    #                     res = None, s = None):
    #     """
        
    #     """

    #     # Start time tracking
    #     startTime = time.perf_counter()

    #     # Initialize thinning procedure
    #     if mode == 'reconstruct' or len(self.ind) == 0:
    #         # Init indices
    #         self.ind = [0]
    #         notIn = list(range(1, self.r.size))     # track non-selected indices 

    #         # Init Newton basis
    #         V = np.zeros((self.r.size, maxIter))
    #         V[:, 0] = (1 / np.sqrt(kernel.normRadon(self.r[0])))\
    #                     * kernel.GramMatrix(self.r, self.a, self.r[0], self.a[0])
            
    #         # Init distance / squared power function
    #         if greedyMethod == 'betaGreedy':
    #             dist = kernel.normRadon(self.r) -  V[:, 0]**2
    #         elif greedyMethod in ['geoDual', 'geoParam', 'geoPeriodic', 'geoSphere']:
    #             dist = km.evalDist(greedyMethod, kernel = kernel, r = self.r,
    #                                 a = self.a, rC = self.r[0], aC = self.a[0], s = s)
            
    #         # Init coefficients and residual
    #         c = np.zeros((maxIter, ))
    #         c[0] = self.Radon[0] / np.sqrt(kernel.normRadon(self.r[0]))
    #         res = self.Radon - c[0] * V[:, 0]

    #         # Set start/stop for greedy selection loop
    #         start = 1
    #         end = maxIter
    #     elif mode == 'sequential':
    #         # Init non-selected indices            
    #         notIn = [index for index in range(self.r.size) if index not in self.ind]

    #         # Init new block for Newton basis
    #         V = np.hstack((V, np.zeros((self.r.size, maxIter))))

    #         # Init new block for coefficients
    #         c = np.hstack((c, np.zeros((maxIter, ))))

    #         # Set start/end for greedy selection loop
    #         start = len(self.ind)
    #         end = start + maxIter

    #     # Greedy selection loop
    #     for k in tqdm(range(start, end), desc = f'{greedyMethod}', 
    #                     bar_format = '{l_bar}{bar:30}{r_bar}',
    #                     total = end, initial = start):
            
    #         # Greedy selection
    #         if greedyMethod == 'betaGreedy':
    #             i = km.betaGreedy(dist[notIn], res[notIn], beta)
    #         elif greedyMethod in ['geoDual', 'geoParam', 'geoPeriodic', 'geoSphere']:
    #             i = np.argmax(dist[notIn])
    #         index = notIn[i]

    #         # Determine power function value at chosen functional
    #         if greedyMethod == 'betaGreedy':
    #             p = np.sqrt(dist[index])
    #         elif greedyMethod in ['geoDual', 'geoParam', 'geoPeriodic', 'geoSphere']:
    #             p = np.sqrt(kernel.normRadon(self.r[index]) - np.sum(V[index, :k]**2))

    #         # Break if power function too small -> numerical instability
    #         if p < tol:
    #             print('Power function too small!')
    #             c = c[:k]
    #             V = V[:, :k]
    #             break

    #         # Update Newton basis
    #         V[notIn, k] = (1/p) * (
    #             kernel.GramMatrix(self.r[notIn], self.a[notIn],
    #                               self.r[index], self.a[index])\
    #             - np.ravel(V[notIn, :k] @ V[[index], :k].T)
    #         )
            
    #         # Update power / distance function
    #         if greedyMethod == 'betaGreedy':
    #             dist[notIn] = dist[notIn] - V[notIn, k]**2
    #         elif greedyMethod in ['geoDual', 'geoParam', 'geoPeriodic', 'geoSphere']:
    #             dist[notIn] = np.minimum(
    #                 dist[notIn],
    #                 km.evalDist(greedyMethod, kernel, self.r[notIn], self.a[notIn],
    #                             self.r[index], self.a[index], s = s)
    #             )

    #         # Determine coefficient of interpolant and update residual
    #         c[k] = res[index] / p
    #         res[notIn] = res[notIn] - c[k] * V[notIn, k]

    #         # Modification of index sets 
    #         self.ind.append(index)
    #         notIn.remove(index)
    #     # end loop
        
    #     # determine elapsed time
    #     endTime = time.perf_counter()
    #     elapsedTime = np.round(endTime - startTime, decimals = 2)
    #     print(5*' ' + '[' + u'\u2713' + ']' + 2*' ' + f'Thinning completed: {elapsedTime} seconds')

    #     # conditional return
    #     if mode == 'sequential':
    #         return c, V, dist, res
    #     return c, V


    # def validationError(self, kernel, rV, aV, cStd, evalVOld, RadonVal):
    #     """
        
    #     """
        
    #     # compute Radon vals on validation parameter set if necessary
    #     if evalVOld is None:
    #         indStart = 0
    #     else: 
    #         indStart = evalVOld.shape[1]

    #     indV = self.ind[indStart:len(self.ind)]

    #     update = kernel.GramMatrix(
    #         rV.reshape((rV.size, 1)), aV.reshape((aV.size, 1)),
    #         self.r[indV].reshape((1, len(indV))),
    #         self.a[indV].reshape((1, len(indV)))
    #     )

    #     if evalVOld is None:
    #         evalVOld = update
    #     else:
    #         evalVOld = np.hstack((evalVOld, update))

    #     interpolant = evalVOld @ cStd.reshape((cStd.size, 1))

    #     interpolant = np.reshape(interpolant, (interpolant.size, ))

    #     return np.sqrt((1/rV.size) * np.sum((RadonVal - interpolant)**2)), evalVOld
    
