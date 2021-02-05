'''
Testing the initial part of HOSSEINsolver with an IDC solver
'''

import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt


def func(t, y):
    return 4*t*(y)**(0.5)


# def y_exact(t):
#     return (1 + t**2)**2


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


def HOSSEINsolverEu(ff, T, y0, N, M):
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
    # d = 1  # len(y0)
    # time step
    h = float(T)/N
    # M: the number of points in calculating quadraure integral
    # (and also the number of steps used in Adam-Bashforth predictor)
    # Note Mm is the number of correctors
    Mm = M - 1
    # Forming the quadraure matrix S[m,i]
    S = np.zeros([Mm, Mm+1])
    for m in range(Mm):  # Calculate qudrature weights
        for i in range(Mm+1):
            x = np.arange(Mm+1)  # Construct a polynomial
            y = np.zeros(Mm+1)   # which equals to 1 at i, 0 at other points
            y[i] = 1
            p = lagrange(x, y)
            para = np.array(p)    # Compute its integral
            P = np.zeros(Mm+2)
            for k in range(Mm+1):
                P[k] = para[k]/(Mm+1-k)
            P = np.poly1d(P)
            S[m, i] = P(m+1) - P(m)
    Svec = S[Mm-1, :]
    # the final answer will be stored in yy
    yy = np.zeros(N+1)
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
    F1 = np.zeros([Mm, M])
    F1[:, 0] = F0
    F2 = F0
    # Y2 [M] new point derived in each level (prediction and corrections)
    Y2 = np.ones(M)*y0
    # ================== INITIAL PART (1) ==================
    # for this part the predictor and correctors step up to M points in time
    # ** predictor ** uses Runge-Kutta 4
    for iTime in range(0, M-1):
        KK1 = F1[0, iTime]
        KK2 = func(t[iTime]+h/2, Y2[0]+KK1*h/2)
        KK3 = func(t[iTime]+h/2, Y2[0]+KK2*h/2)
        KK4 = func(t[iTime]+h,   Y2[0]+KK3*h)
        Y2[0] = Y2[0] + h*(KK1 + 2*KK2 + 2*KK3 + KK4)/6
        # Y2[0] = Y2[0] + h*KK1
        F1[0, iTime+1] = func(t[iTime+1], Y2[0])
    # ** correctors ** use Integral Deffered Correction
    for iCor in range(1, M-1):
        ll = iCor - 1
        for iTime in range(0, M-1):
            Y2[iCor] = Y2[iCor] + h*(F1[iCor, iTime]-F1[ll, iTime]) + \
                h * np.dot(S[iTime, :], F1[ll, :])
            F1[iCor, iTime+1] = func(t[iTime+1], Y2[iCor])
    # treat the last correction loop a little different
    for iTime in range(0, M-1):
        Y2[M-1] = Y2[M-1] + h*(F2-F1[M-2, iTime]) + \
            h * np.dot(S[iTime, :], F1[M-2, :])
        F2 = func(t[iTime+1], Y2[M-1])
        yy[iTime+1] = Y2[M-1]

    # ================== INITIAL PART (2) ==================
    beta_vec = beta(M)
    for iTime in range(M-1, 2*M-2):
        iStep = iTime - (M-1)
        # prediction loop
        Y2[0] = Y2[0] + h*np.dot(beta_vec, F1[0, :])
        # correction loops
        for ll in range(iStep):
            iCor = ll + 1
            Y2[iCor] = Y2[iCor] + h*(F1[iCor, -1]-F1[ll, -2]) + \
                h * np.dot(Svec, F1[ll, :])
        F1[0, 0:M-1] = F1[0, 1:M]
        F1[0, M-1] = func(t_ext[iTime+1], Y2[0])
        for ll in range(iStep):
            iCor = ll + 1
            F1[iCor, 0:M-1] = F1[iCor, 1:M]
            F1[iCor, M-1] = func(t_ext[iTime+1-iCor], Y2[iCor])

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
            F1[ll, M-1] = func(t_ext[iTime+1-ll], Y2[ll])
        # storing the final answer
        yy[iTime+1-(M-1)] = Y2[M-1]
        F2 = func(t_ext[iTime+1-(M-1)], Y2[M-1])
        # ---> updating predictor stencil
        # ** approach #0:
        F1[0, 0:M-1] = F1[0, 1:M]
        # ** approach #1: pushing the most correct answer to predictor
        # F1[0, 0] = F2
        # F1[0, 1:M-1] = F1[0, 2:M]
        # ** approach #2 : pushing the recently corrected answer of
        # each corrector to the associated node in predictor
        # F1[0, 0] = F2
        # for ii in range(1, M-1):
        #     F1[0, ii] = F1[-ii, -1]

        F1[0, M-1] = func(t_ext[iTime+1], Y2[0])

    return t, yy


def HOSSEINsolverAB(ff, T, y0, N, M):
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
    # d = 1  # len(y0)
    # time step
    h = float(T)/N
    # M: the number of points in calculating quadraure integral
    # (and also the number of steps used in Adam-Bashforth predictor)
    # Note Mm is the number of correctors
    Mm = M - 1
    # Forming the quadraure matrix S[m,i]
    S = np.zeros([Mm, Mm+1])
    for m in range(Mm):  # Calculate qudrature weights
        for i in range(Mm+1):
            x = np.arange(Mm+1)  # Construct a polynomial
            y = np.zeros(Mm+1)   # which equals to 1 at i, 0 at other points
            y[i] = 1
            p = lagrange(x, y)
            para = np.array(p)    # Compute its integral
            P = np.zeros(Mm+2)
            for k in range(Mm+1):
                P[k] = para[k]/(Mm+1-k)
            P = np.poly1d(P)
            S[m, i] = P(m+1) - P(m)
    Svec = S[Mm-1, :]
    # the final answer will be stored in yy
    yy = np.zeros(N+1)
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
    F1 = np.zeros([Mm, M])
    F1[:, 0] = F0
    F2 = F0
    # Y2 [M] new point derived in each level (prediction and corrections)
    Y2 = np.ones(M)*y0
    # ================== INITIAL PART (1) ==================
    # for this part the predictor and correctors step up to M points in time
    # ** predictor ** uses Runge-Kutta 4
    for iTime in range(0, M-1):
        KK1 = F1[0, iTime]
        KK2 = func(t[iTime]+h/2, Y2[0]+KK1*h/2)
        KK3 = func(t[iTime]+h/2, Y2[0]+KK2*h/2)
        KK4 = func(t[iTime]+h,   Y2[0]+KK3*h)
        Y2[0] = Y2[0] + h*(KK1 + 2*KK2 + 2*KK3 + KK4)/6
        # Y2[0] = Y2[0] + h*KK1
        F1[0, iTime+1] = func(t[iTime+1], Y2[0])
    # ** correctors ** use Integral Deffered Correction
    for iCor in range(1, M-1):
        ll = iCor - 1
        for iTime in range(0, M-1):
            Y2[iCor] = Y2[iCor] + h*(F1[iCor, iTime]-F1[ll, iTime]) + \
                h * np.dot(S[iTime, :], F1[ll, :])
            F1[iCor, iTime+1] = func(t[iTime+1], Y2[iCor])
    # treat the last correction loop a little different
    for iTime in range(0, M-1):
        Y2[M-1] = Y2[M-1] + h*(F2-F1[M-2, iTime]) + \
            h * np.dot(S[iTime, :], F1[M-2, :])
        F2 = func(t[iTime+1], Y2[M-1])
        yy[iTime+1] = Y2[M-1]

    # ================== INITIAL PART (2) ==================
    beta_vec = beta(M)
    beta_vec2 = beta(M-1)
    for iTime in range(M-1, 2*M-2):
        iStep = iTime - (M-1)
        # prediction loop
        Y2[0] = Y2[0] + h*np.dot(beta_vec, F1[0, :])
        # correction loops
        for ll in range(iStep):
            iCor = ll + 1
            Y2[iCor] = Y2[iCor] + h*(F1[iCor, -1]-F1[ll, -2]) + \
                h * np.dot(Svec, F1[ll, :])
        F1[0, 0: M-1] = F1[0, 1: M]
        F1[0, M-1] = func(t_ext[iTime+1], Y2[0])
        for ll in range(iStep):
            iCor = ll + 1
            F1[iCor, 0: M-1] = F1[iCor, 1: M]
            F1[iCor, M-1] = func(t_ext[iTime+1-iCor], Y2[iCor])

    # ================== MAIN LOOP FOR TIME ==================
    for iTime in range(2*M-2, N+M-1):
        # prediction loop
        Y2[0] = Y2[0] + h*np.dot(beta_vec, F1[0, :])
        # correction loops up to the second last one
        for ll in range(M-2):
            iCor = ll + 1
            # Y2[iCor] = Y2[iCor] + h*(F1[iCor, -1]-F1[ll, -2]) + \
            #    h * np.dot(Svec, F1[ll, :])
            Fvec = np.array([F1[iCor, -3]-F1[ll, -4], F1[iCor, -2] -
                             F1[ll, -3], F1[iCor, -1]-F1[ll, -2]])
            Y2[iCor] = Y2[iCor] + h*np.dot(beta_vec2, Fvec) + \
                h * np.dot(Svec, F1[ll, :])
        # last correction loop
        F2m = func(t_ext[iTime+1-(M-1)-2], yy[iTime+1-(M-1)-2])
        F2mm = func(t_ext[iTime+1-(M-1)-3], yy[iTime+1-(M-1)-3])
        Fvec = np.array([F2mm-F1[M-2, -4], F2m-F1[M-2, -3], F2-F1[M-2, -2]])
        Y2[M-1] = Y2[M-1] + h*np.dot(beta_vec2, Fvec) + \
            h * np.dot(Svec, F1[M-2, :])

        # ~~~~~~~~~~~ Updating Stencil ~~~~~~~~~~~
        # ---> updating correctors stencil
        for ll in range(1, M-1):
            F1[ll, 0: M-1] = F1[ll, 1: M]
            F1[ll, M-1] = func(t_ext[iTime+1-ll], Y2[ll])
        # storing the final answer
        yy[iTime+1-(M-1)] = Y2[M-1]
        F2 = func(t_ext[iTime+1-(M-1)], Y2[M-1])
        # ---> updating predictor stencil
        # ** approach #0:
        F1[0, 0: M-1] = F1[0, 1: M]
        # ** approach #1: pushing the most correct answer to predictor
        # F1[0, 0] = F2
        # F1[0, 1:M-1] = F1[0, 2:M]
        # ** approach #2 : pushing the recently corrected answer of
        # each corrector to the associated node in predictor
        # F1[0, 0] = F2
        # for ii in range(1, M-1):
        #     F1[0, ii] = F1[-ii, -1]

        F1[0, M-1] = func(t_ext[iTime+1], Y2[0])

    return t, yy


T1 = 96
M = 4
Mm = M - 1
y0 = 1
# Y_exact = (1 + np.linspace(0, T1, N)**2)**2
yExact = (1 + T1**2)**2
rangeN = [int(10**n) for n in np.arange(1.8, 3.8, 0.2)]
rangeNpower = [(2*10**15)*int(10**n)**(-10) for n in np.arange(1.8, 3.8, 0.2)]
rangeNpower2 = [(3*10**13)*int(10**n)**(-9) for n in np.arange(1.8, 3.8, 0.2)]

err0 = np.empty(np.shape(rangeN))
err2 = np.empty(np.shape(rangeN))
for i, NN in enumerate(rangeN):
    t_0, yy_0 = HOSSEINsolverEu(func, T1, y0, NN-1, M)
    t_2, yy_2 = HOSSEINsolverAB(func, T1, y0, NN-1, M)
    err0[i] = abs((yExact-yy_0[-1])/yExact)
    err2[i] = abs((yExact-yy_2[-1])/yExact)

fig, ax = plt.subplots()
ax.plot(rangeN, err0, label='Euler')
ax.plot(rangeN, err2, label='Pred AB4, Corr AB3')
ax.plot(rangeN, rangeNpower, label='e = N^{-10}')
ax.plot(rangeN, rangeNpower2, label='e = N^{-9}')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('N')  # Add an x-label to the axes.
ax.set_ylabel('$|y - y_{exact}|$')
ax.set_title("Errors using different methods")
ax.legend(loc='lower left')
plt.show()
# print(Y_exact[-1])
# print(yy_0[-1], abs(Y_exact[-1]-yy_0[-1]))
# print(yy_2[-1], abs(Y_exact[-1]-yy_2[-1]))
# print('error ratio', abs(Y_exact[-1]-yy_2[-1])/abs(Y_exact[-1]-yy_0[-1]))
