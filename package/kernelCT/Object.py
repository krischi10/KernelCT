import numpy as np
from scipy.linalg import solve, solve_triangular
from scipy.sparse.linalg import lsmr
import kernelCT.kernelMethods as km
from tqdm import tqdm
import kernelCT.phantom as phantom
import kernelCT.fbp as fbp
import time

class Object:
    """
    
    """
    
    def __init__(self, type, X, Y, r, a, ind = None, eval = None, Radon = None):
        """
        
        """
        
        self.type = type    # object type
        self.X = X          # x-coordinates of domain
        self.Y = Y          # y-coordinates of domain
        self.r = r          # radial coordinates of Radon samples
        self.a = a          # angular coordinates of Radon samples

        # Init indices of data points used for reconstruction
        if ind is None:
            self.ind = []
        else:
            self.ind = ind

        # Compute point evaluations and Radon data of object on attributed domain
        if self.type == 'unknown':
            self.eval = eval
            self.Radon = Radon
        else:
            self.phantomEval()


    def phantomEval(self):
        """
        
        """

        self.eval = phantom.getPhantomEval(self.type, self.X, self.Y)
        self.Radon = phantom.getPhantomRadon(self.type, self.r, self.a)



    def reconstructFBP(self, filter, spline, rMax, RMax):
        """
        
        """

        # compute parameters via optimal sampling equalities
        fWidth = self.a.shape[0] / rMax         # filter width
        d = np.pi / fWidth                      # sampling rate for radial variable
        convM = int((self.a.shape[1]-1)/2) \
                    + int(np.ceil(RMax/d))      # radius of convolution filter
        lenData = int(np.ceil(RMax/d))          # number of spline interpolation points

        # setup data for spline interpolation
        xEval = np.cos(self.a[:, [0]]) * self.X.reshape((1, self.X.size)) \
                + np.sin(self.a[:, [0]]) * self.Y.reshape((1, self.Y.size))
        xData = d * np.arange( -lenData, lenData + 1)

        # compute low pass filtering + backproject spline interpolation
        reconstruction = fbp.backProjection(
            fbp.rowSplines(xEval, xData,
                fbp.convRows(self.Radon,
                    d * fbp.FourierLP(filter, fWidth, convM)
                ),
                spline
            )
        )

        return np.reshape(reconstruction, self.X.shape)




    def reconstructKernel(self, kernel, method, gamma = None, cNewton = None,
                          V = None, returnC = False, evalROld = None):
        """
        
        """
        
        # Compute coefficients of interpolant in terms of standard basis
        print('Compute coefficients of interpolant in terms of standard basis...')
        start = time.perf_counter()

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

        end = time.perf_counter()
        elapsed_time = np.round(end - start, decimals = 2)
        print(5*' ' + '[' + u'\u2713' + ']' + 2*' ' + f'Coefficients computed: {elapsed_time} seconds')

        # Construct interpolant via Riesz representers
        print('Eval representers...')
        start = time.perf_counter()


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
        
        end = time.perf_counter()
        elapsed_time = np.round(end - start, decimals = 2)
        print(5*' ' + '[' + u'\u2713' + ']' + 2*' ' + f'Representers evaluated: {elapsed_time} seconds')

        # Check if coefficients should be returned
        if returnC:
            return np.reshape(reconstruction, self.X.shape), cStd, evalROld
        return np.reshape(reconstruction, self.X.shape)



    def kernelThinning(self, kernel, greedyMethod, maxIter, mode = 'reconstruct',
                        beta = None, tol = 1e-15, c = None, V = None, dist = None,
                        res = None, s = None):
        """
        
        """

        # Start time tracking
        startTime = time.perf_counter()

        # Initialize thinning procedure
        if mode == 'reconstruct' or len(self.ind) == 0:
            # Init indices
            self.ind = [0]
            notIn = list(range(1, self.r.size))     # track non-selected indices 

            # Init Newton basis
            V = np.zeros((self.r.size, maxIter))
            V[:, 0] = (1 / np.sqrt(kernel.normRadon(self.r[0])))\
                        * kernel.GramMatrix(self.r, self.a, self.r[0], self.a[0])
            
            # Init distance / squared power function
            if greedyMethod == 'betaGreedy':
                dist = kernel.normRadon(self.r) -  V[:, 0]**2
            elif greedyMethod in ['geoDual', 'geoParam', 'geoPeriodic', 'geoSphere']:
                dist = km.evalDist(greedyMethod, kernel = kernel, r = self.r,
                                    a = self.a, rC = self.r[0], aC = self.a[0], s = s)
            
            # Init coefficients and residual
            c = np.zeros((maxIter, ))
            c[0] = self.Radon[0] / np.sqrt(kernel.normRadon(self.r[0]))
            res = self.Radon - c[0] * V[:, 0]

            # Set start/stop for greedy selection loop
            start = 1
            end = maxIter
        elif mode == 'sequential':
            # Init non-selected indices            
            notIn = [index for index in range(self.r.size) if index not in self.ind]

            # Init new block for Newton basis
            V = np.hstack((V, np.zeros((self.r.size, maxIter))))

            # Init new block for coefficients
            c = np.hstack((c, np.zeros((maxIter, ))))

            # Set start/end for greedy selection loop
            start = len(self.ind)
            end = start + maxIter

        # Greedy selection loop
        for k in tqdm(range(start, end), desc = f'{greedyMethod}', 
                        bar_format = '{l_bar}{bar:30}{r_bar}',
                        total = end, initial = start):
            
            # Greedy selection
            if greedyMethod == 'betaGreedy':
                i = km.betaGreedy(dist[notIn], res[notIn], beta)
            elif greedyMethod in ['geoDual', 'geoParam', 'geoPeriodic', 'geoSphere']:
                i = np.argmax(dist[notIn])
            index = notIn[i]

            # Determine power function value at chosen functional
            if greedyMethod == 'betaGreedy':
                p = np.sqrt(dist[index])
            elif greedyMethod in ['geoDual', 'geoParam', 'geoPeriodic', 'geoSphere']:
                p = np.sqrt(kernel.normRadon(self.r[index]) - np.sum(V[index, :k]**2))

            # Break if power function too small -> numerical instability
            if p < tol:
                print('Power function too small!')
                c = c[:k]
                V = V[:, :k]
                break

            # Update Newton basis
            V[notIn, k] = (1/p) * (
                kernel.GramMatrix(self.r[notIn], self.a[notIn],
                                  self.r[index], self.a[index])\
                - np.ravel(V[notIn, :k] @ V[[index], :k].T)
            )
            
            # Update power / distance function
            if greedyMethod == 'betaGreedy':
                dist[notIn] = dist[notIn] - V[notIn, k]**2
            elif greedyMethod in ['geoDual', 'geoParam', 'geoPeriodic', 'geoSphere']:
                dist[notIn] = np.minimum(
                    dist[notIn],
                    km.evalDist(greedyMethod, kernel, self.r[notIn], self.a[notIn],
                                self.r[index], self.a[index], s = s)
                )

            # Determine coefficient of interpolant and update residual
            c[k] = res[index] / p
            res[notIn] = res[notIn] - c[k] * V[notIn, k]

            # Modification of index sets 
            self.ind.append(index)
            notIn.remove(index)
        # end loop
        
        # determine elapsed time
        endTime = time.perf_counter()
        elapsedTime = np.round(endTime - startTime, decimals = 2)
        print(5*' ' + '[' + u'\u2713' + ']' + 2*' ' + f'Thinning completed: {elapsedTime} seconds')

        # conditional return
        if mode == 'sequential':
            return c, V, dist, res
        return c, V


    def validationError(self, kernel, rV, aV, cStd, evalVOld, RadonVal):
        """
        
        """
        
        # compute Radon vals on validation parameter set if necessary
        if evalVOld is None:
            indStart = 0
        else: 
            indStart = evalVOld.shape[1]

        indV = self.ind[indStart:len(self.ind)]

        update = kernel.GramMatrix(
            rV.reshape((rV.size, 1)), aV.reshape((aV.size, 1)),
            self.r[indV].reshape((1, len(indV))),
            self.a[indV].reshape((1, len(indV)))
        )

        if evalVOld is None:
            evalVOld = update
        else:
            evalVOld = np.hstack((evalVOld, update))

        interpolant = evalVOld @ cStd.reshape((cStd.size, 1))

        interpolant = np.reshape(interpolant, (interpolant.size, ))

        return np.sqrt((1/rV.size) * np.sum((RadonVal - interpolant)**2)), evalVOld
    
