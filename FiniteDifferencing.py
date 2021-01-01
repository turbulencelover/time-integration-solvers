import numpy as np
import time

class FiniteDifferencing:
    def dirchlet_central(self, alpha, f, boundary_conds, a, b, h):
        n_increments = int((b-a)/h)
        
        # u(t,x)
        u = np.zeros([n_increments+1, n_increments+1])
        xs = np.linspace(a,b, n_increments+1)

        # initial condition u(t=0,x)=f(x)
        u[0] = [f(x) for x in xs]

        for t in range(n_increments):
            u[t+1] = [boundary_conds[0]] \
                        + [u[t,n] + alpha**2*(h/(u[t,n]-u[t,n-1]))
                            *(u[t,n+1]-2*u[t,n]+ u[t,n-1]) for n in range(1, n_increments)]\
                        + [boundary_conds[1]]
        return u

    def neumann_central(self, alpha, f, boundary_conds, a, b, h):
        n_increments = int((b-a)/h)
        
        # u(t,x)
        u = np.zeros([n_increments+1, n_increments+1])
        xs = np.linspace(a,b, n_increments+1)
        
        # initial condition u(t=0,x)=f(x)
        u[0] = [f(x) for x in xs]
        
        for t in range(n_increments):
            u[t+1]=[boundary_conds]+[u[t,n] + 2*(alpha**2)*(h/(u[t,n]-u[t,n-1]))*(u[t,n-1]-u[t,n]) for n in range(1,n_increments+1)]

        return u

if __name__ == '__main__':
    # neuman (only when du_dx=0)

    ics = lambda x : x*(1-x)
    alpha =1
    bcs_d= [0,0]
    bcs_n = 0
    a=0
    b=1
    h=0.1
    u = FiniteDifferencing().dirchlet_central(#.neumann_central(
            alpha=alpha,
            f=ics,
            boundary_conds=bcs_d,
            a=a,
            b=b, 
            h=h)
    print(u)