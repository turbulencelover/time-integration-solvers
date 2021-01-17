'''
Testing the initial part of HOSSEINsolver with an IDC solver
'''

import numpy as np
fimport matplotlib.pyplot as plt
import time
from scipy.integrate import quadrature
from functools import reduce


def func0(t, y):
    return 4*t*(y)**(0.5)


def y_exact0(t):
    return (1 + t**2)**2


def func1(t, y):
    y1 = y[0]
    y2 = y[1]
    y1_p = t*y2 + y1  # -y2 + y1*(1-y1**2-y2**2)
    y2_p = -t*y1 + y2  # y1 + 3*y2*(1-y1**2-y2**2)
    return np.array([y1_p, y2_p])


def y_exact1(ts):
    def y1(t): return np.exp(t) * (np.cos(0.5*t**2)+np.sin(0.5*t**2))
    def y2(t): return np.exp(t) * (np.cos(0.5*t**2)-np.sin(0.5*t**2))
    return np.array([[y1(t) for t in ts], [y2(t) for t in ts]])


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


def solver(func, T, y0, N, M, approach):
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

        F1[0, M-1] = func(t_ext[iTime+1], Y2[0])

    return t, yy


def test(T, y0, M, N, approach, ff, y_t):
    start = time.perf_counter()
    tt, yy = solver(ff, T, y0, N, M, approach)
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')

    ts = np.linspace(0, T, N+1)
    y_true = y_t(ts)
    return tt, yy.T, y_true


# tt, yy, y_true = test(T=1, y0=np.array(
#     [1.0, 1.0]), M=4, N=1000, approach=2, ff=func1, y_t=y_exact1)

# print(yy[0].shape)

# plt.plot(tt, yy[0], tt, y_true[0])

# plt.plot(tt, yy[1], tt, y_true[1])


# T1 = 96
# M = 4
# Mm = M - 1
# y0 = 1
# # Y_exact = (1 + np.linspace(0, T1, N)**2)**2
# yExact = (1 + T1**2)**2
rangeN = [int(10**n) for n in np.arange(1.8, 3.8, 0.2)]
rangeNpower = [(10**12)*int(10**n)**(-9) for n in np.arange(1.8, 3.8, 0.2)]

err00, err01 = [], []
err10, err11 = [], [] #np.empty(np.shape((2,rangeN)))
err20, err21 = [], [] #np.empty(np.shape((2,rangeN)))

for i, NN in enumerate(rangeN):
    t0, y_p0, y_t0 = test(T=1, y0=np.array(
        [1.0, 1.0]), M=4, N=NN-1, approach=0, ff=func1, y_t=y_exact1)
    t1, y_p1, y_t1 = test(T=1, y0=np.array(
        [1.0, 1.0]), M=4, N=NN-1, approach=1, ff=func1, y_t=y_exact1)
    t2, y_p2, y_t2 = test(T=1, y0=np.array(
        [1.0, 1.0]), M=4, N=NN-1, approach=1, ff=func1, y_t=y_exact1)

    x,y =abs((y_t0-y_p0)/y_t0)
    err00.append(np.mean(x))
    err01.append(y[-1])#np.mean(y))

    x,y = abs((y_t1-y_p1)/y_t1)
    err10.append(np.mean(x))
    err11.append(y[-1])#np.mean(y))

    x,y = abs((y_t2-y_p2)/y_t2)
    err20.append(np.mean(x))
    err21.append(y[-1])#np.mean(y))


print(err00)


fig, ax = plt.subplots()
ax.plot(rangeN, err01, label='No feedback')
ax.plot(rangeN, err11, label='1')
ax.plot(rangeN, err21, label='With feedback')
ax.plot(rangeN, rangeNpower, label='e = N^{-9}')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('N')  # Add an x-label to the axes.
ax.set_ylabel('Average |y - y_{exact}|')
ax.set_title("Errors using different methods")
ax.legend(loc='lower left')
plt.show()
# print(Y_exact[-1])
# print(yy_0[-1], abs(Y_exact[-1]-yy_0[-1]))
# print(yy_2[-1], abs(Y_exact[-1]-yy_2[-1]))
# print('error ratio', abs(Y_exact[-1]-yy_2[-1])/abs(Y_exact[-1]-yy_0[-1]))
