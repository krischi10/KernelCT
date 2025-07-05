
class KernelInterpolant:
     pass

class KernelRegressor:
     pass

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