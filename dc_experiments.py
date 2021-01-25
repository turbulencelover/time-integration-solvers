'''
Analysing RIDC parameters for predictor and correctors
'''

from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
from functools import reduce
from scipy.integrate import quadrature
import numpy as np
from functools import wraps
from time import time


class DCs:

    def timing(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time()
            result = f(*args, **kwargs)
            end = time()
            return result, end-start
        return wrapper

    @timing
    def idc_fe(self, a, b, alpha, N, p, f):
        """Perform IDCp-FE
        Input: (a,b) endpoints; alpha ics; N #intervals; p order; f vector field.
        Require: N divisible by M=p-1, with JM=N. M is #corrections.
        Return: eta_sol
        """

        # Initialise, J intervals of size M
        if not type(N) is int:
            raise TypeError('N must be integer')
        M = p-1
        if N % M != 0:
            raise Exception('p-1 does not divide N')
        dt = (b-a)/N
        J = int(N/M)
        S = np.zeros([M, M+1])

        # M corrections, M intervals I, of size J
        eta_sol = np.zeros(N+1)
        eta_sol[0] = alpha
        eta = np.zeros([M+1, J, M+1])
        t = np.zeros([J, M+1])
        eta_overlaps = np.zeros([J])

        # Precompute integration matrix
        for m in range(M):
            for i in range(M+1):
                def c(t, i): return reduce(lambda x, y: x*y,
                                           [(t-k)/(i-k) for k in range(M) if k != i])
                S[m, i] = quadrature(c, m, m+1, args=(i))[0]

        for j in range(J):
            # Prediction Loop
            eta[0, j, 0] = alpha if j == 0 else eta_overlaps[j]
            for m in range(M):
                t[j, m] = (j*M+m)*dt
                eta[0, j, m+1] = eta[0, j, m] + dt*f(t[j, m], eta[0, j, m])

            # Correction Loops
            for l in range(1, M+1):
                eta[l, j, 0] = eta[l-1, j, 0]
                for m in range(M):
                    # Error equation, Forward Euler
                    term1 = dt*(f(t[j, m], eta[l, j, m]) -
                                f(t[j, m], eta[l-1, j, m]))
                    term2 = dt * \
                        np.sum([S[m, i] * f(t[j, i], eta[l-1, j, i])
                                for i in range(M)])
                    eta[l, j, m+1] = eta[l, j, m] + term1 + term2

            eta_sol[j*M+1:(j+1)*M + 1] = eta[M, j, 1:]
            if j != J-1:
                eta_overlaps[j+1] = eta[M, j, M]

        return eta_sol[:-1]

    @timing
    def ridc_fe(self, a, b, alpha, N, p, K, f):
        """Perform IDCp-FE
        Input: (a,b) endpoints; alpha ics; N #intervals; p order; K intervals; f vector field.
        # corrections. J groups of K intervals.
        Require: N divisible by K, with JK=N. M is
        Return: eta_sol
        """

        # Initialise, J intervals of size M
        if not isinstance(N, int):
            raise TypeError('N must be integer')
        M = p-1
        if N % K != 0:
            raise Exception('K does not divide N')
        dt = (b-a)/N
        J = int(N/K)
        S = np.zeros([M, M+1])

        # M corrections, M intervals I, of size J
        eta_sol = np.zeros(N+1)
        eta_sol[0] = alpha
        eta = np.zeros([M+1, J, K+1])
        t = np.zeros([J, K+1])
        eta_overlaps = np.zeros([J])

        # Precompute integration matrix
        for m in range(M):
            for i in range(M+1):
                def c(t, i): return reduce(lambda x, y: x*y,
                                           [(t-k)/(i-k) for k in range(M) if k != i])
                S[m, i] = quadrature(c, m, m+1, args=(i))[0]

        for j in range(J):
            # Prediction Loop
            eta[0, j, 0] = alpha if j == 0 else eta_overlaps[j]
            for m in range(K):
                t[j, m] = (j*K+m)*dt
                eta[0, j, m+1] = eta[0, j, m] + dt*f(t[j, m], eta[0, j, m])

            # Correction Loops
            for l in range(1, M+1):
                eta[l, j, 0] = eta[l-1, j, 0]
                for m in range(M):
                    # Error equation, Forward Euler
                    term1 = dt*(f(t[j, m], eta[l, j, m]) -
                                f(t[j, m], eta[l-1, j, m]))
                    term2 = dt*np.sum([S[m, i] * f(t[j, i], eta[l-1, j, i])
                                       for i in range(M)])
                    eta[l, j, m+1] = eta[l, j, m] + term1 + term2
                for m in range(M, K):
                    term1 = dt*(f(t[j, m], eta[l, j, m]) -
                                f(t[j, m], eta[l-1, j, m]))
                    term2 = dt * \
                        np.sum([S[M-1, i] * f(t[j, m-M+i], eta[l-1, j, m-M+i])
                                for i in range(M)])
                    eta[l, j, m+1] = eta[l, j, m] + term1 + term2

            eta_sol[j*K+1:(j+1)*K + 1] = eta[M, j, 1:]
            if j != J-1:
                eta_overlaps[j+1] = eta[M, j, K]

        return eta_sol

    @timing
    def ridc_rk4(self, a, b, alpha, N, p, K, f):
        """Perform RIDC(p,K)-RK(4)
        Input: (a,b) endpoints; alpha ics; N #intervals; p order; K intervals; f vector field.
        # corrections. J groups of K intervals.
        Require: N divisible by K, with JK=M
        Return: eta_sol
        """

        # Initialise, J intervals of size M
        if not isinstance(N, int):
            raise TypeError('N must be integer')
        M = p-1
        if N % K != 0:
            raise Exception('K does not divide N')
        dt = (b-a)/N
        J = int(N/K)
        S = np.zeros([M, M+1])

        # M corrections, M intervals I, of size J
        eta_sol = np.zeros(N+1)
        eta_sol[0] = alpha
        eta = np.zeros([M+1, J, K+1])
        t = np.zeros([J, K+1])
        eta_overlaps = np.zeros([J])

        # Precompute integration matrix
        for m in range(M):
            for i in range(M+1):
                def c(t, i): return reduce(lambda x, y: x*y,
                                           [(t-k)/(i-k) for k in range(M) if k != i])
                S[m, i] = quadrature(c, m, m+1, args=(i))[0]

        for j in range(J):
            # Prediction Loop
            eta[0, j, 0] = alpha if j == 0 else eta_overlaps[j]
            for m in range(K):
                t[j, m] = (j*K+m)*dt
                k1 = f(t[j, m], eta[0, j, m])
                k2 = f(t[j, m]+dt/2, eta[0, j, m] + k1/2)
                k3 = f(t[j, m]+dt/2, eta[0, j, m]+k2/2)
                k4 = f(t[j, m]+dt, eta[0, j, m]+k3)
                eta[0, j, m+1] = eta[0, j, m] + dt*(k1/6 + k2/3 + k3/3 + k4/6)

            # Correction Loops
            for l in range(1, M+1):
                eta[l, j, 0] = eta[l-1, j, 0]
                for m in range(M):
                    # Error equation, Forward Euler
                    term1 = dt*(f(t[j, m], eta[l, j, m]) -
                                f(t[j, m], eta[l-1, j, m]))
                    term2 = dt*np.sum([S[m, i] * f(t[j, i], eta[l-1, j, i])
                                       for i in range(M)])
                    eta[l, j, m+1] = eta[l, j, m] + term1 + term2
                for m in range(M, K):
                    term1 = dt*(f(t[j, m], eta[l, j, m]) -
                                f(t[j, m], eta[l-1, j, m]))
                    term2 = dt * \
                        np.sum([S[M-1, i] * f(t[j, m-M+i], eta[l-1, j, m-M+i])
                                for i in range(M)])
                    eta[l, j, m+1] = eta[l, j, m] + term1 + term2

            eta_sol[j*K+1:(j+1)*K + 1] = eta[M, j, 1:]
            if j != J-1:
                eta_overlaps[j+1] = eta[M, j, K]

        return eta_sol

    @timing
    def ridc_ab2(self, a, b, alpha, N, p, K, f):
        """Perform RIDC(p,K)-AB2
        Input: (a,b) endpoints; alpha ics; N #intervals; p order; K intervals; f vector field.
        # corrections. J groups of K intervals.
        Require: N divisible by K, with JK=N. M is
        Return: eta_sol
        """
        # Initialise, J intervals of size M
        if not isinstance(N, int):
            raise TypeError('N must be integer')
        M = p-1
        if N % K != 0:
            raise Exception('K does not divide N')
        dt = (b-a)/N
        J = int(N/K)
        S = np.zeros([M, M+1])

        # M corrections, M intervals I, of size J
        eta_sol = np.zeros(N+1)
        eta_sol[0] = alpha
        eta = np.zeros([M+1, J, K+1])
        t = np.zeros([J, K+1])
        eta_overlaps = np.zeros([J])

        # Precompute integration matrix
        for m in range(M):
            for i in range(M+1):
                def c(t, i): return reduce(lambda x, y: x*y,
                                           [(t-k)/(i-k) for k in range(M) if k != i])
                S[m, i] = quadrature(c, m, m+1, args=(i))[0]

        for j in range(J):
            # Prediction Loop ADAM BASHFORTH
            eta[0, j, 0] = alpha if j == 0 else eta_overlaps[j]
            eta[0, j, 1] = eta[0, j, 0] + dt*f(t[j, 0], eta[0, j, 0])
            t[j, 0], t[j, 1] = j*K*dt, (j*K+1)*dt
            for m in range(K-1):
                t[j, m+1] = (j*K+m+1)*dt
                eta[0, j, m+2] = eta[0, j, m+1] \
                    + 1.5*dt*f(t[j, m+1], eta[0, j, m+1]) \
                    - 0.5*dt*f(t[0, m], eta[0, j, m])

            # Correction Loops
            for l in range(1, M+1):
                eta[l, j, 0] = eta[l-1, j, 0]
                for m in range(M):
                    # Error equation, Forward Euler
                    term1 = dt*(f(t[j, m], eta[l, j, m]) -
                                f(t[j, m], eta[l-1, j, m]))
                    term2 = dt*np.sum([S[m, i] * f(t[j, i], eta[l-1, j, i])
                                       for i in range(M)])
                    eta[l, j, m+1] = eta[l, j, m] + term1 + term2
                for m in range(M, K):
                    term1 = dt*(f(t[j, m], eta[l, j, m]) -
                                f(t[j, m], eta[l-1, j, m]))
                    term2 = dt * \
                        np.sum([S[M-1, i] * f(t[j, m-M+i], eta[l-1, j, m-M+i])
                                for i in range(M)])
                    eta[l, j, m+1] = eta[l, j, m] + term1 + term2

            eta_sol[j*K+1:(j+1)*K + 1] = eta[M, j, 1:]
            if j != J-1:
                eta_overlaps[j+1] = eta[M, j, K]

        return eta_sol

    @timing
    def ridc_abM(self, T, y0, N, M, approach, f):
        '''
        Inputs:
        ff: the RHS of the system of ODEs y'=f(t,y)
        T:  integration interval[0,T]
        y0: initial condition
        N:  number of nodes
        M: the number of points in calculating quadraure integral
        (and also the number of steps used in Adam-Bashforth predictor)
        or number of correction loops PLUS the prection loop

        Output:
        t: time vector
        yy: solution as a function of time
        '''

        # number of equations in ODE (aka degree of freedom, dimension of space)
        # for now set to 1 (will be modified LATER to handle more than one dime)
        d = len(y0)
        # time step
        h = float(T)/N
        # M: the number of points in calculating quadraure integral
        # (and also the number of steps used in Adam-Bashforth predictor)
        # Note Mm is the number of correctors
        Mm = M - 1
        # Forming the quadraure matrix S[m,i]
        S = np.zeros([Mm, Mm+1])
        for m in range(Mm):
            for i in range(Mm+1):
                def c(t, i): return reduce(lambda x, y: x*y,
                                           [(t-k)/(i-k) for k in range(M) if k != i])
                S[m, i] = quadrature(c, m, m+1, args=(i))[0]
        Svec = S[Mm-1]
        # the final answer will be stored in yy
        yy = np.zeros([N+1, d])
        # putting the initial condition in y
        yy[0] = y0
        # Value of RHS at initial time
        F0 = f(0, y0)
        # the time vector
        t = np.arange(0, T+h, h)
        # extended time vector (temporary: cuz I didn't write code for end part)
        t_ext = np.arange(0, T+h+M*h, h)
        # F vector and matrice:
        # the RHS of ODE is evaluated and stored in this vector and matrix:
        # F1 [M x M]: first index is the order (0=prection, 1=first correction)
        # second index is the time (iTime)
        # Note F1 could have been [M-1 x M] as the first two rows are equal to each
        # other BUT we designed it as a place holder for future parallelisation
        F1 = np.zeros([Mm, M, d])
        F1[:, 0] = F0
        F2 = F0
        # Y2 [M] new point derived in each level (prediction and corrections)
        Y2 = np.ones([M, d])*y0
        # ================== INITIAL PART (1) ==================
        # for this part the predictor and correctors step up to M points in time
        # ** predictor ** uses Runge-Kutta 4
        for iTime in range(0, M-1):
            KK1 = F1[0, iTime]
            KK2 = f(t[iTime]+h/2, Y2[0]+KK1*h/2)
            KK3 = f(t[iTime]+h/2, Y2[0]+KK2*h/2)
            KK4 = f(t[iTime]+h,   Y2[0]+KK3*h)
            Y2[0] = Y2[0] + h*(KK1 + 2*KK2 + 2*KK3 + KK4)/6
            F1[0, iTime+1] = f(t[iTime+1], Y2[0])
        # ** correctors ** use Integral Deffered Correction
        for iCor in range(1, M-1):
            ll = iCor - 1
            for iTime in range(0, M-1):
                Y2[iCor] = Y2[iCor] + h*(F1[iCor, iTime]-F1[ll, iTime]) + \
                    h * np.dot(S[iTime], F1[ll])
                F1[iCor, iTime+1] = f(t[iTime+1], Y2[iCor])
        # treat the last correction loop a little different
        for iTime in range(0, M-1):
            Y2[M-1] = Y2[M-1] + h*(F2-F1[M-2, iTime]) + \
                h * np.dot(S[iTime], F1[M-2])
            F2 = f(t[iTime+1], Y2[M-1])
            yy[iTime+1] = Y2[M-1]

        # ================== INITIAL PART (2) ==================
        def beta(M):
            '''
            Generates beta coefficients for Adam-Bashforth integrating scheme
            These coefficients are stored in reversed compared to conventional
            Adam-Bashforth implementations (the first element of beta corresponds to
            earlier point in time).
            input:
            M: the order of Adam-Bashforth scheme
            '''

            if M == 2:
                return np.array([-1./2, 3./2])
            elif M == 3:
                return np.array([5./12, -16./12, 23./12])
            elif M == 4:
                return np.array([-9./24, 37./24, -59./24, 55./24])
            elif M == 5:
                return np.array([251./720, -1274./720, 2616./720, -2774./720, 1901./720])
            elif M == 6:
                return np.array([-475./720, 2877./720, -7298./720, 9982./720, -7923./720, 4277./720])

        beta_vec = beta(M)
        for iTime in range(M-1, 2*M-2):
            iStep = iTime - (M-1)
            # prediction loop
            Y2[0] = Y2[0] + h*np.dot(beta_vec, F1[0])
            # correction loops
            for ll in range(iStep):
                iCor = ll + 1
                Y2[iCor] = Y2[iCor] + h*(F1[iCor, -1]-F1[ll, -2]) + \
                    h * np.dot(Svec, F1[ll])
            F1[0, 0:M-1] = F1[0, 1:M]
            F1[0, M-1] = f(t_ext[iTime+1], Y2[0])
            for ll in range(iStep):
                iCor = ll + 1
                F1[iCor, 0:M-1] = F1[iCor, 1:M]
                F1[iCor, M-1] = f(t_ext[iTime+1-iCor], Y2[iCor])

        # ================== MAIN LOOP FOR TIME ==================
        for iTime in range(2*M-2, N+M-1):
            # prediction loop
            Y2[0] = Y2[0] + h*np.dot(beta_vec, F1[0, :])
            # correction loops up to the second last one
            for ll in range(M-2):
                iCor = ll + 1
                Y2[iCor] = Y2[iCor] + h*(F1[iCor, -1]-F1[ll, -2]) + \
                    h * np.dot(Svec, F1[ll, :])
            # last correction loop
            Y2[M-1] = Y2[M-1] + h * (F2-F1[M-2, -2]) + \
                h * np.dot(Svec, F1[M-2, :])

            # ~~~~~~~~~~~ Updating Stencil ~~~~~~~~~~~
            # ---> updating correctors stencil
            for ll in range(1, M-1):
                F1[ll, 0:M-1] = F1[ll, 1:M]
                F1[ll, M-1] = f(t_ext[iTime+1-ll], Y2[ll])
            # storing the final answer
            yy[iTime+1-(M-1)] = Y2[M-1]
            F2 = f(t_ext[iTime+1-(M-1)], Y2[M-1])
            # ** updating predictor stencil
            if (approach == 0):
                F1[0, 0:M-1] = F1[0, 1:M]
            # ** approach #1: pushing the most correct answer to predictor
            elif (approach == 1):
                F1[0, 0] = F2
                F1[0, 1:M-1] = F1[0, 2:M]
            # ** approach #2 : pushing the recently corrected answer of
            else:
                # each corrector to the associated node in predictor
                F1[0, 0] = F2
                for ii in range(1, M-1):
                    F1[0, ii] = F1[-ii, -1]

            F1[0, M-1] = f(t_ext[iTime+1], Y2[0])

        return t, yy

    @timing
    def ridc_rk4_ab4(self, func, T, y0, N, M):
        '''
        Inputs:
        ff: the RHS of the system of ODEs y'=f(t,y)
        T:  integration interval[0,T]
        y0: initial condition
        N:  number of nodes
        M: the number of points in calculating quadraure integral
        (and also the number of steps used in Adam-Bashforth predictor)
        or number of correction loops PLUS the prection loop

        Output:
        t: time vector
        yy: solution as a function of time
        '''
        # define Adam Bashforth four step explicit method
        # considering placement of stencil
        def corrector(y1, y0, t, h, f):
            e1 = f(t[0], y1[0]) - f(t[0], y0[0])
            e2 = f(t[1], y1[1]) - f(t[1], y0[1])
            e3 = f(t[2], y1[2]) - f(t[2], y0[2])
            e4 = f(t[3], y1[3]) - f(t[3], y0[3])
            return y1[3] + h/24 * (55*e4 - 59*e3 + 37*e2 - 9*e1)

        def predictor(y, t, h, f):
            k1 = f(t[0], y[0])
            k2 = f(t[1], y[1])
            k3 = f(t[2], y[2])
            k4 = f(t[3], y[3])  # most recent
            return y[3] + h/24 * (55*k4 - 59*k3 + 37*k2 - 9*k1)

        def euler_predictor(y, t, h, f): return y+h*f(t, y)
        def euler_corrector(y1, y0, t, h, f): return y1 + h*(f(t, y1)-f(t, y0))

        # number of equations in ODE (aka degree of freedom, dimension of space)
        # for now set to 1 (will be modified LATER to handle more than one dime)
        d = len(y0)
        # time step
        h = float(T)/N
        # M: the number of points in calculating quadraure integral
        # (and also the number of steps used in Adam-Bashforth predictor)
        # Note Mm is the number of correctors
        Mm = M - 1
        # Forming the quadraure matrix S[m,i]
        S = np.zeros([Mm, Mm+1])
        for m in range(Mm):
            for i in range(Mm+1):
                def c(t, i): return reduce(lambda x, y: x*y,
                                           [(t-k)/(i-k) for k in range(M) if k != i])
                S[m, i] = quadrature(c, m, m+1, args=(i))[0]
        Svec = S[Mm-1]
        # the final answer will be stored in yy
        yy = np.zeros([N+1, d])
        # putting the initial condition in y
        yy[0] = y0
        # Value of RHS at initial time
        F0 = func(0, y0)
        # the time vector
        t = np.arange(0, T+h, h)
        # extended time vector (temporary: cuz I didn't write code for end part)
        t_ext = np.arange(0, T+h+M*h, h)
        # F vector and matrice:
        # the RHS of ODE is evaluated and stored in this vector and matrix:
        # F1 [M x M]: first index is the order (0=prection, 1=first correction)
        # second index is the time (iTime)
        # Note F1 could have been [M-1 x M] as the first two rows are equal to each
        # other BUT we designed it as a place holder for future parallelisation
        F1 = np.zeros([Mm, M, d])
        F1[:, 0] = F0
        F2 = F0
        # Y2 [M] new point derived in each level (prediction and corrections)
        Y2 = np.ones([M, d])*y0
        # ================== INITIAL PART (1) ==================
        # for this part the predictor and correctors step up to M points in time
        # ** predictor ** uses Runge-Kutta 4
        for iTime in range(0, M-1):
            KK1 = F1[0, iTime]
            KK2 = func(t[iTime]+h/2, Y2[0]+KK1*h/2)
            KK3 = func(t[iTime]+h/2, Y2[0]+KK2*h/2)
            KK4 = func(t[iTime]+h,   Y2[0]+KK3*h)
            Y2[0] = Y2[0] + h*(KK1 + 2*KK2 + 2*KK3 + KK4)/6
            F1[0, iTime+1] = func(t[iTime+1], Y2[0])
        # ** correctors ** use Integral Deffered Correction
        for iCor in range(1, M-1):
            ll = iCor - 1
            for iTime in range(0, M-1):
                Y2[iCor] = Y2[iCor] + h*(F1[iCor, iTime]-F1[ll, iTime]) + \
                    h * np.dot(S[iTime], F1[ll])
                F1[iCor, iTime+1] = func(t[iTime+1], Y2[iCor])
        # treat the last correction loop a little different
        for iTime in range(0, M-1):
            Y2[M-1] = Y2[M-1] + h*(F2-F1[M-2, iTime]) + \
                h * np.dot(S[iTime], F1[M-2])
            F2 = func(t[iTime+1], Y2[M-1])
            yy[iTime+1] = Y2[M-1]

        # ================== INITIAL PART (2) ==================
        def beta(M):
            '''
            Generates beta coefficients for Adam-Bashforth integrating scheme
            These coefficients are stored in reversed compared to conventional
            Adam-Bashforth implementations (the first element of beta corresponds to
            earlier point in time).
            input:
            M: the order of Adam-Bashforth scheme
            '''

            if M == 2:
                return np.array([-1./2, 3./2])
            elif M == 3:
                return np.array([5./12, -16./12, 23./12])
            elif M == 4:
                return np.array([-9./24, 37./24, -59./24, 55./24])
            elif M == 5:
                return np.array([251./720, -1274./720, 2616./720, -2774./720, 1901./720])
            elif M == 6:
                return np.array([-475./720, 2877./720, -7298./720, 9982./720, -7923./720, 4277./720])
        beta_vec = beta(M)
        for iTime in range(M-1, 2*M-2):
            iStep = iTime - (M-1)
            # * prediction loop AB4
            Y2[0] = Y2[0] + h*np.dot(beta_vec, F1[0])
            # correction loops
            for ll in range(iStep):
                iCor = ll + 1
                Y2[iCor] = Y2[iCor] + h*(F1[iCor, -1]-F1[ll, -2]) + \
                    h * np.dot(Svec, F1[ll])
            F1[0, 0:M-1] = F1[0, 1:M]
            F1[0, M-1] = func(t_ext[iTime+1], Y2[0])
            for ll in range(iStep):
                iCor = ll + 1
                F1[iCor, 0:M-1] = F1[iCor, 1:M]
                F1[iCor, M-1] = func(t_ext[iTime+1-iCor], Y2[iCor])

        # ================== MAIN LOOP FOR TIME ==================
        for iTime in range(2*M-2, N+M-1):
            # * prediction loop AB
            Y2[0] = Y2[0] + h*np.dot(beta_vec, F1[0, :])
            # ! correction loops up to the second last one
            # * AB correction
            for ll in range(M-2):
                iCor = ll + 1
                Y2[iCor] = Y2[iCor] + h*(F1[iCor, -1]-F1[ll, -2]) + \
                    h * np.dot(Svec, F1[ll, :])

            # last correction loop
            Y2[M-1] = Y2[M-1] + h * (F2-F1[M-2, -2]) + \
                h * np.dot(Svec, F1[M-2, :])

            # ~~~~~~~~~~~ Updating Stencil ~~~~~~~~~~~
            # ---> updating correctors stencil
            for ll in range(1, M-1):
                F1[ll, 0:M-1] = F1[ll, 1:M]
                F1[ll, M-1] = func(t_ext[iTime+1-ll], Y2[ll])
            # storing the final answer
            yy[iTime+1-(M-1)] = Y2[M-1]
            F2 = func(t_ext[iTime+1-(M-1)], Y2[M-1])
            # ** updating predictor stencil
            F1[0, 0:M-1] = F1[0, 1:M]

            F1[0, M-1] = func(t_ext[iTime+1], Y2[0])

        return t, yy
