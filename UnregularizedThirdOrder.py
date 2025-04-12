import numpy as np
import cvxpy as cvx
import numpy.linalg as LA
from AdaptiveFramework import *

def min3(H,Q,b):
    n = len(b)

    constraints = []
    Y = cvx.Variable((n,n), symmetric=True)
    y = cvx.Variable((n,1))
    z = cvx.Variable((1,1))
    
    
    Hess = Q + sum([H[i,:,:]*y[i] for i in range(n)])
    L = cvx.Variable((n,1))
    
    constraints = constraints + [L[i] == cvx.trace(H[i,:,:]@Y) + Q[i,:]@y for i in range(n)]
    constraints = constraints + [Q[i,:]@y + cvx.trace(H[i,:,:]@Y)/2 + b[i] == 0 for i in range(n)]
    
    T = cvx.bmat([[Hess, L], [L.T, z]])
    YYT = cvx.bmat([[Y, y], [y.T, np.identity(1)]])
    constraints = constraints + [T >> 0]
    constraints = constraints + [YYT >> 0]
    
    prob = cvx.Problem(cvx.Minimize(cvx.trace(Q@Y)/2 + b.T@y + z/2),constraints)
    prob.solve(solver = cvx.SCS, verbose=False)
    
    px = prob.value
    return list([y.value,px,prob.status])

def newton_run(fx, dx, d2x, d3x, x0, max_iterations, tol):
    k = 0
    x_curr = np.reshape(x0, (2, 1))
    converged = False
    
    try:
        while k < max_iterations:
            grad = dx(x_curr)
            if LA.norm(grad) <= tol:
                converged = True
                break

            hessian = d2x(x_curr) + 0.0001 * np.identity(2)
            inv_hessian = LA.inv(hessian)
            x_curr = x_curr - inv_hessian @ grad
            k += 1
        
        if not converged:
            converged = (LA.norm(dx(x_curr)) <= tol)
    
    except LA.LinAlgError:
        converged = False
    
    return x_curr, converged
            
def unregularized_third_newton_run(fx, dx, d2x, d3x, x0, max_iterations, tol):
    k = 0
    x_curr = np.reshape(x0, (2, 1))
    
    while k < max_iterations and LA.norm(dx(x_curr)) > tol and LA.norm(x_curr) < 25:
        D3x = d3x(x_curr)
        D2x = d2x(x_curr)
        Dx = dx(x_curr)
        out = min3(D3x, D2x, Dx)

        if 'infeasible' in out[2] or 'unbounded' in out[2]:
            alpha = alpha_approx(Dx, D2x, D3x)
            out = min3(D3x, D2x + alpha*np.eye(2), Dx)
        
        x_curr = x_curr + out[0]
        k += 1
    
    converged = (LA.norm(dx(x_curr)) <= tol)
    
    return x_curr, converged