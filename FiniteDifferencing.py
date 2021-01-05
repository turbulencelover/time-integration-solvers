import numpy as np
import time


class Heat_fd:
    def heat1d_fe(self, mu, f, boundary_conds, X, T, N, J):
        a, b = X
        dt = (T[1]-T[0])/N
        dx = (b-a)/J

        # u(t,x)
        u = np.zeros([N+1, J+1])
        xs = np.arange(a, b + dx, dx)

        # initial condition u(t=0,x)=f(x)
        u[0] = [f(x) for x in xs]
        
        u[:,0], u[:,-1] = boundary_conds
        
        l = mu*(dt/dx**2)

        for t in range(N):
            u[t+1, 1:J] = [u[t, n] + l*(u[t, n+1]-2*u[t, n] + u[t, n-1])
                          for n in range(1, J)]

        return u


if __name__ == '__main__':
    # neuman (only when du_dx=0)

    def ics(x): return x*(1-x)
    mu = 1
    bcs = (0,0)
    X= (0,1)
    J = 10
    N = 10
    T = (0,5)
    u = Heat_fd().heat1d_fe(
        mu=mu,
        f=ics,
        boundary_conds=bcs,
        X=X,
        T=T,
        N=N,
        J=J)

    print(u)
