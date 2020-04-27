import numpy as np
import scipy as sp
import osqp

from scipy import sparse


class OptimizationProblem:

    def __init__(self, T, obj_steps):
        
        """
        T: number of time-slots
        obj_steps: number of breaks in objective function

        A: basic matrix structure
        u: upper bound of the problem
        l: lower bounds of the problem



        eff_c, eff_d: charging / discharing efficiency in [0,1]
        dmax: positive, maximum charging of bat
        dmin: positive, maximum discharing of bat
        bmax: maximum battery capacity
        bmin: minimum battery capcaity
        charge: current battery chage:

        load: load or forcast of the load in the T time-slots.
        cost: objective function of the minimization problem.

        >> bmin <= charge <= bmax

        obj: the objective coefficients. entry (o, t) is the coefficient of the o-th break at time-slot t
        """
        self.T = T
        self.obj_steps = obj_steps

        self.A = self.__build_matrix(T, obj_steps)
        self.A_sparse = sparse.csc_matrix(self.A)
        self.eff_c = 1.0
        self.eff_d = 1.0
        self.obj_slopes = np.ones((obj_steps, T)) * 1.0
        self.obj_offsets = np.zeros((obj_steps, T))

        L = self.A.shape[0]

        self.charge = 0
        self.bmax = 0
        self.bmin = 0
        self.dmax = 0
        self.dmin = 0

        self.load = np.zeros(T)
        self.cost = np.array([0] * (2 * T) + [1] * T)

        self.P = sparse.csc_matrix(np.zeros((3 * T, 3 * T)))

        l = self._build_l()
        u = self._build_u()
        self.prob = osqp.OSQP()
        self.prob.setup(P=self.P, q=self.cost, A=self.A_sparse, l=l, u=u, polish=True, eps_abs=1e-6, eps_rel=1e-6)


    def __build_matrix(self, T, obj_steps):

        CI = np.tril(np.ones((T, T)))
        DI = - np.tril(np.ones((T, T)))
        I  = np.eye(T)
        Z = np.zeros((T, T))

        block = [
            [CI, DI, Z],
            [I, Z, Z],
            [Z, I, Z]
        ]
        block.extend([[I, -I, -I] for _ in range(obj_steps)])

        A = np.block(block)
        return A

    def _build_l(self):
        """
        Build the lower bound of the problem
        """
        N, T = self.A.shape[0], self.T
        l = np.zeros(N)
        l[: T] = self.bmin - self.charge
        l[T : T * 2] = 0 # Redundant
        l[T * 2: T * 3] = 0 # Redundant
        l[T * 3:] = - np.infty
        return l

    def _build_u(self):
        """
        Build the lower bound of the problem
        """
        N, T = self.A.shape[0], self.T
        u = np.zeros(N)
        u[: T] = self.bmax - self.charge
        u[T : T * 2] = self.dmax 
        u[T * 2: T * 3] = self.dmin

        P = (self.obj_slopes * self.load).flatten(order='C')
        u[T * 3:] = - P 
        return u

    def set_effc(self, ec):
        """
        Sets the charging efficiency
        """

        x_, y_ = self.T * 3, self.T
        self.A[x_ :, : y_] *= self.eff_c
        self.A[x_ :, : y_] /= ec
        self.eff_c = ec
        self.A_sparse = sparse.csc_matrix(self.A)

    def set_effd(self, ed):
        """
        Sets the charging efficiency
        """

        x_, y_, y_end = self.T * 3, self.T, self.T * 2
        self.A[x_ :, y_ : y_end] /= self.eff_d
        self.A[x_ :, y_ : y_end] *= ed
        self.eff_d = ed
        self.A_sparse = sparse.csc_matrix(self.A)


    def set_dmax(self, dmax): self.dmax = dmax
    def set_dmin(self, dmin): self.dmin = dmin
    def set_bmax(self, bmax): self.bmax = bmax
    def set_bmin(self, bmin): self.bmin = bmin
    def set_charge(self, charge): self.charge = charge

    def set_load(self, values, index=None):
        """
        Updates load with values. If index is not none,
        use those as the index to fill in
        """
        arr = index if index is not None else np.arange(self.T)
        arr = np.array(arr)
        values = np.array(values)
        np.put(self.load, arr, values)

    def set_prices_slope(self, step, values, index=None):
        """
        Sets the prices for the `step` breakpoint
        of the objective function (counted) from left to right.
        If index is not None, updates only those values, otherwise,
        update them all.
        """
        assert step <= self.obj_steps
        arr = index if index is not None else np.arange(self.T)
        arr = np.array(arr)
        values = np.array(values)
        L = arr.shape[0]

        T_ = self.T * (3 + step)
        T = self.T
        A = self.A
        A[T_ : T_ + T, : 2 * T][arr] /= self.obj_slopes[step, arr].reshape((L, -1))
        A[T_ : T_ + T, : 2 * T][arr] *= values.reshape((L, -1))
        
        np.put(self.obj_slopes[step], arr, values)
        self.A_sparse = sparse.csc_matrix(self.A)
        
    def set_prices_offset(self, step, values, index=None):
        """
        Sets the prices for the `step` breakpoint
        of the objective function (counted) from left to right.
        If index is not None, updates only those values, otherwise,
        update them all.
        """
        assert step <= self.obj_steps
        arr = index if index is not None else np.arange(self.T)
        arr = np.array(arr)
        values = np.array(values)
        L = arr.shape[0]

        T_ = self.T * (3 + step)
        T = self.T
        A = self.A
        A[T_ : T_ + T, : 2 * T][arr] /= self.obj_slopes[step, arr].reshape((L, -1))
        A[T_ : T_ + T, : 2 * T][arr] *= values.reshape((L, -1))
        
        np.put(self.obj_slopes[step], arr, values)
        self.A_sparse = sparse.csc_matrix(self.A)

    def optimize(self):

        l_ = self._build_l()
        u_ = self._build_u()
        self.prob.update(l=l_, u=u_, Ax=self.A_sparse.data)
        res = self.prob.solve()
        return res



    

