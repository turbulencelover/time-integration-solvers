import time
import numpy as np
from scipy.interpolate import lagrange
from scipy.integrate import quadrature
from functools import reduce

# Example function
dy_dt = lambda t, y : 4*t*np.sqrt(y)
y = lambda t: y0*(1+t**2)**2
y0=1

# define Adam Bashforth four step explicit method
# considering placement of stencil
def corrector(y1, y0, t, h, f):
    e1 = f(t[0], y1[0]) - f(t[0], y0[0])
    e2 = f(t[1], y1[1]) - f(t[1], y0[1])
    e3 = f(t[2], y1[2]) - f(t[2], y0[2])
    e4 = f(t[3], y1[3]) - f(t[3], y0[3])
    return y1[3] + h/24 * (55*e4 - 59*e3 + 37*e2 - 9*e1)
    

def predictor(y,t, h, f):
    k1 = f(t[0], y[0])
    k2 = f(t[1], y[1])
    k3 = f(t[2], y[2]) 
    k4 = f(t[3], y[3]) # most recent
    return y[3] + h/24 * (55*k4 - 59*k3 + 37*k2 - 9*k1)

euler_predictor = lambda y,t,h,f : y+h*f(t,y)
euler_corrector = lambda y1,y0,t,h,f : y1 + h*(f(t,y1)-f(t,y0))


def iridc_ab(a,b,alpha, N, p, K, f):
    """Perform IDCp-FE
    Input: (a,b) endpoints; alpha ics; N #intervals; p order; K intervals; f vector field.
    Require: N divisible by K, with JK=N. M is #corrections. J groups of K intervals.
    Return: eta_sol
    """

    # Initialise, J intervals of size M 
    if not isinstance(N, int):
        raise TypeError('N must be integer')
    M = p-1
    if N % K !=0:
        raise Exception('K does not divide N')
    J = int(N/K)
    h = float(T)/N
    # number of equations in ODE (aka degree of freedom, dimension of space)
    #d = len(y0)
    S = np.zeros([M, M+1])

    for m in range(M):
        for i in range(M+1):
            c = lambda t,i : reduce(lambda x, y: x*y, 
                                [(t-k)/(i-k) for k in range(M) if k!=i])
            S[m,i] = quadrature(c, m, m+1, args=(i))[0] 

    # ! Svec = S[M-1, :]
    # storing the final answer in yy
    eta_sol = np.zeros([N+1])
    # the time vector
    t = np.arange(0, T+h, h)
    t_ext = np.arange(0, T+h+M*h, h)
    # putting the initial condition in eta_sol
    eta_sol[0] = y0
    # loop over each group of intervals j
    for j in range(J):   
        print('Interval, ', j)
        # ----------------------------------------------------------------------
        # Phase 1: compute to the point every threads can start simultaneously
        eta_begin = np.zeros([M+1, 2*M])
        # predictor starts with last point in j-1 interval
        eta_begin[0, 0] = eta_sol[j*K]

        # predictor loop using forward Euler method
        for m in range(3):
            t[m+1] = (j*K+m+1) * h
            eta_begin[0, m+1] = euler_predictor(eta_begin[0, m], t[j*K+m], h, f)
        for m in range(2*M-1):
            t[m+1] = (j*K+m+1) * h
            eta_begin[0, m+1] = predictor(eta_begin[0,m-3:m+1], t[j*K+m-3:j*K+m+1], h, f)

        # corrector loops using Lagrange polynomials
        for l in range(1, M+1):
            eta_begin[l,:] = np.zeros([2*M])
            eta_begin[l, 0] = eta_sol[j*K]
            for m in range(3):
                eta_begin[l,m+1] = euler_corrector(eta_begin[l,m], eta_begin[l-1,m], t[j*K+m], h, f)\
                    + h*sum([S[m, i]*f(t[j*K+i], eta_begin[l-1, i]) for i in range(M+1)])
            for m in range(3,M):
                eta_begin[l, m+1] = corrector(eta_begin[l,m-3:m+1], eta_begin[l-1,m-3:m+1], t[j*K+m-3:j*K+m+1], h, f)\
                    + h*sum([S[m, i]*f(t[j*K+i], eta_begin[l-1, i]) for i in range(M+1)])
            for m in range(M, 2*M-l):
                eta_begin[l, m+1] = corrector(eta_begin[l,m-3:m+1], eta_begin[l-1,m-3:m+1], t[j*K+m-3:j*K+m+1], h, f)\
                    + h*sum([S[M-1, i]*f(t[j*K+m-M+i+1], eta_begin[l-1, m-M+i+1]) for i in range(M+1)])

        eta_pred = eta_begin[0, -4:]
        eta_sol[j*K:j*K+M] = eta_begin[M, :M]

        # * eta0 is previous correction level
        # * eta1 is current correction
        eta0 = np.zeros([M, M+1])
        eta1 = np.zeros([M])
        for l in range(1, M+1):
            # 'lm' is for corrector 'l' (trying to save space for Y1corr&Y2corr
            lm = l - 1
            eta0[lm, :] = eta_begin[l-1, M-l:2*M-l+1]
            eta1[lm] = eta_begin[l, 2*M-l-1]

        # ----------------------------------------------------------------------
        # Phase 2: all threads can go simultaneously now
        for m in range(M-1, K):
            # predictor
            # Tpred = t_ext[j*K+m+M]
            # Ypred = predictor(Ypred, Tpred)
            # ? Using AB-4 what does t_ext actually stand for
            ts_pred = t_ext[j*K+M+m-3 : j*K+M+m+1]
            eta_pred = predictor(eta_pred, ts_pred)


            # correctors
            for l in range(1, M+1):
                lm = l - 1
                # ! dimension of stencil is M (used for the regression) but
                # ! the euler correction only uses value -2 in Hosseins code 
                Tvec = t_ext[j*K+m-l+1:j*K+m-l+1+M+1]
                eta1[lm] = corrector(eta1[lm], eta0[lm, :], Tvec)
            
            # * update the stencil
            eta0[0, :M] = eta0[0, 1:M+1]
            eta0[:, 0, M] = eta_pred
            for lm in range(1, M):
                # TODO : stencil is different? Check Compare1.py?
                eta0[lm, :M] = eta0[lm, 1:M+1]
                eta0[lm, M] = eta1[lm-1]
            # put the most corrected point in the final answer
            eta_sol[j*K+m+1] = eta1[M-1]


    return t, eta_sol

T = 10.0
p = 4  # RIDC(6,40)
K = 200
N = 1000
a = 0

start = time.perf_counter()
tt, yy = iridc_ab(a=a, b=T, alpha=y0, N=N, p=p, K=K, f=dy_dt)
finish = time.perf_counter()
print(f'Finished in {round(finish-start, 2)} second(s)')
print(tt[-1:])
print(yy[:, -1])