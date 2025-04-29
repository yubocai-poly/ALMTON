import numpy as np
import cvxpy as cvx
import numpy.linalg as LA
from src.AdaptiveFramework import *


def min3(H, Q, b):
    n = len(b)

    constraints = []
    Y = cvx.Variable((n, n), symmetric=True)
    y = cvx.Variable((n, 1))
    z = cvx.Variable((1, 1))

    Hess = Q + sum([H[i, :, :] * y[i] for i in range(n)])
    L = cvx.Variable((n, 1))

    constraints = constraints + [
        L[i] == cvx.trace(H[i, :, :] @ Y) + Q[i, :] @ y for i in range(n)
    ]
    constraints = constraints + [
        Q[i, :] @ y + cvx.trace(H[i, :, :] @ Y) / 2 + b[i] == 0 for i in range(n)
    ]

    T = cvx.bmat([[Hess, L], [L.T, z]])
    YYT = cvx.bmat([[Y, y], [y.T, np.identity(1)]])
    constraints = constraints + [T >> 0]
    constraints = constraints + [YYT >> 0]

    prob = cvx.Problem(
        cvx.Minimize(cvx.trace(Q @ Y) / 2 + b.T @ y + z / 2), constraints
    )
    prob.solve(solver=cvx.SCS, verbose=False)

    px = prob.value
    return list([y.value, px, prob.status])


def newton_run(fx, dx, d2x, d3x, x0, max_iterations, tol):
    k = 0
    x_curr = np.reshape(x0, (2, 1))
    converged = False

    try:
        while k < max_iterations:
            k += 1
            grad = dx(x_curr)
            if LA.norm(grad) <= tol:
                converged = True
                break

            hessian = d2x(x_curr)
            inv_hessian = LA.inv(hessian)
            x_curr = x_curr - inv_hessian @ grad

        if not converged:
            converged = LA.norm(dx(x_curr)) <= tol

    except LA.LinAlgError:
        converged = False

    return x_curr, converged, k


def gradient_descent(f, df, x0, alpha, max_iter, tol):
    x = x0.copy()
    converged = False
    k = 0
    for k in range(max_iter):
        grad = df(x)
        x_new = x - alpha * grad
        if LA.norm(x_new - x) < tol:
            converged = True
            break
        x = x_new
    return x, converged, k + 1


def unregularized_third_newton_run(fx, dx, d2x, d3x, x0, max_iterations, tol):
    k = 0
    x_curr = np.reshape(x0, (2, 1))
    converged = False

    while k < max_iterations:
        if LA.norm(dx(x_curr)) <= tol or LA.norm(x_curr) >= 25:
            break

        try:
            D3x = d3x(x_curr)
            D2x = d2x(x_curr)
            Dx = dx(x_curr)
            out = min3(D3x, D2x, Dx)

            if "infeasible" in out[2] or "unbounded" in out[2]:
                alpha = alpha_approx(Dx, D2x, D3x)
                out_regularized = min3(D3x, D2x + alpha * np.eye(2), Dx)
                if out_regularized[0] is not None:
                    out = out_regularized
                else:
                    break  # failure

            if out[0] is not None:
                x_curr += out[0]
                k += 1
            else:
                break

        except Exception as e:
            print(f"Iteration {k} failed: {str(e)}")
            break

    converged = LA.norm(dx(x_curr)) <= tol and LA.norm(x_curr) < 25
    return x_curr, converged, k
