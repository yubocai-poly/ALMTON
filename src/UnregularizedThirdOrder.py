import numpy as np
import cvxpy as cvx
import numpy.linalg as LA
from src.AdaptiveFramework import *

"""
Implementation of the Unregularized Third-Order Newton Method.

Adapted from the codebase accompanying:
    Olha Silina and Jeffrey Zhang, "An Unregularized Third Order Newton Method," 2023. (https://arxiv.org/abs/2209.10051)
    https://github.com/jeffreyzhang92/Third_Order_Newton/blob/main/Newton_Fractal.py

Modifications: Built upon the original codebase by adding imports and support for different solvers.
"""

# Check solver availability once at module import time
_MOSEK_AVAILABLE = False
_CVXOPT_AVAILABLE = False

try:
    import mosek

    _MOSEK_AVAILABLE = True
except ImportError:
    pass

try:
    import cvxopt

    _CVXOPT_AVAILABLE = True
except ImportError:
    pass


def min3(H, Q, b, solver="auto", sdp_tol=1e-5): 
    """
    Solve the SDP subproblem for unregularized third-order Newton method.

    Parameters:
    - H: Third-order tensor (n, n, n)
    - Q: Hessian matrix (n, n)
    - b: Gradient vector (n, 1)
    - solver: Solver to use ('auto', 'mosek', 'scs', 'cvxopt')
              'auto' (default): tries MOSEK first if available, falls back to SCS
    - sdp_tol: Tolerance for SDP solver (default: 1e-5)
               For inexact Newton: use lower tolerance (e.g., 1e-3) in early iterations,
               higher tolerance (e.g., 1e-6) near convergence

    Returns:
    - list: [y.value, prob.value, prob.status]
    """
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

    # Determine solver list (sorted by priority)
    solver_list = []

    if solver == "auto":
        # Smart selection: prioritize MOSEK, fall back to SCS
        if _MOSEK_AVAILABLE:
            solver_list.append(("mosek", cvx.MOSEK, {}))
        solver_list.append(
            (
                "scs",
                cvx.SCS,
                {
                    "eps": sdp_tol,
                    "max_iters": 10000,
                    "scale": 10.0,
                    "normalize": True,
                },
            )
        )
        # Add CVXOPT as a final fallback if available
        if _CVXOPT_AVAILABLE:
            solver_list.append(("cvxopt", cvx.CVXOPT, {}))

    elif solver == "mosek":
        if _MOSEK_AVAILABLE:
            solver_list = [("mosek", cvx.MOSEK, {})]
        else:
            # If MOSEK is requested but not available, fall back to SCS with warning
            print("Warning: MOSEK requested but not available, falling back to SCS.")
            solver_list = [
                (
                    "scs",
                    cvx.SCS,
                    {"eps": sdp_tol, "max_iters": 10000, "scale": 10.0, "normalize": True},
                )
            ]

    elif solver == "scs":
        solver_list = [
            (
                "scs",
                cvx.SCS,
                {"eps": sdp_tol, "max_iters": 10000, "scale": 10.0, "normalize": True},
            )
        ]

    elif solver == "cvxopt":
        if _CVXOPT_AVAILABLE:
            solver_list = [("cvxopt", cvx.CVXOPT, {})]
        else:
            print("Warning: CVXOPT requested but not available, falling back to SCS.")
            solver_list = [
                (
                    "scs",
                    cvx.SCS,
                    {"eps": sdp_tol, "max_iters": 10000, "scale": 10.0, "normalize": True},
                )
            ]

    else:
        # Default fallback
        solver_list = [
            (
                "scs",
                cvx.SCS,
                {"eps": sdp_tol, "max_iters": 10000, "scale": 10.0, "normalize": True},
            )
        ]

    last_status = "solver_not_run"

    for solver_name, solver_obj, solver_params in solver_list:
        try:
            prob.solve(solver=solver_obj, verbose=False, **solver_params)

            # Check if solve status is successful
            # Note: cvxpy status is not only OPTIMAL, sometimes OPTIMAL_INACCURATE is also acceptable
            if prob.status in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
                # Ensure y.value is not None (SCS sometimes returns optimal status but None value on failure)
                if y.value is not None:
                    px = prob.value
                    return list([y.value, px, prob.status])

            # Record the last status for debugging
            last_status = prob.status

        except Exception as e:
            # Catch exceptions thrown by solver (e.g., CVXOPT KKT matrix singularity error)
            # print(f"Solver {solver_name} failed with error: {e}") # Optional: print debug info
            last_status = f"{solver_name}_exception"
            continue  # Try next solver

    # If all solvers fail, return failure flag
    # Upper-level code should check for 'solver_failure' or 'infeasible' status
    return list(
        [
            None,
            np.inf,
            last_status if last_status != "solver_not_run" else "solver_failure",
        ]
    )


def backtracking_line_search(  
    f, df, x, p, alpha_init=1.0, c=1e-4, rho=0.5, max_ls_iter=50
):
    """
    Backtracking line search using Armijo condition.

    Parameters:
    - f: Objective function
    - df: Gradient function
    - x: Current point
    - p: Search direction
    - alpha_init: Initial step size (default: 1.0)
    - c: Armijo parameter (default: 1e-4)
    - rho: Backtracking factor (default: 0.5)
    - max_ls_iter: Maximum line search iterations (default: 50)

    Returns:
    - alpha: Step size
    """
    alpha = alpha_init
    f_x = f(x)
    grad_x = df(x)
    grad_dot_p = grad_x.flatten() @ p.flatten()

    for _ in range(max_ls_iter):
        x_new = x + alpha * p
        f_new = f(x_new)

        # Armijo condition: f(x + αp) ≤ f(x) + c*α*∇f(x)'*p
        if f_new <= f_x + c * alpha * grad_dot_p:
            return alpha

        alpha = rho * alpha

    # If line search fails, return a very small step
    return alpha


def gradient_descent( 
    f, df, x0, alpha_init=1.0, max_iter=100, tol=1e-6, use_line_search=True
):
    """
    Gradient descent with optional backtracking line search.

    Parameters:
    - f: Objective function
    - df: Gradient function
    - x0: Initial point
    - alpha_init: Initial step size (used if use_line_search=False, or as initial guess for line search)
    - max_iter: Maximum iterations
    - tol: Convergence tolerance (gradient norm)
    - use_line_search: Whether to use backtracking line search (default: True)

    Returns:
    - x: Final point
    - converged: Boolean indicating convergence
    - k: Number of iterations
    """
    x = x0.copy().reshape(-1, 1)
    converged = False
    k = 0

    for k in range(max_iter):
        grad = df(x)
        grad_norm = LA.norm(grad)

        # Check convergence
        if grad_norm <= tol:
            converged = True
            break

        # Compute search direction (negative gradient)
        p = -grad

        # Determine step size
        if use_line_search:
            # Use backtracking line search
            alpha = backtracking_line_search(f, df, x, p, alpha_init=alpha_init)
        else:
            # Use fixed step size
            alpha = alpha_init

        # Update
        x_new = x + alpha * p

        # Check for stagnation
        if LA.norm(x_new - x) < tol * 1e-3:
            converged = True
            break

        x = x_new

    return x, converged, k + 1


def newton_run( 
    fx,
    dx,
    d2x,
    d3x,
    x0,
    max_iterations,
    tol,
    use_line_search=True,
    min_eigenvalue_tol=1e-6,
):
    """
    Newton's method with optional line search and saddle point detection.

    Parameters:
    - fx: Objective function
    - dx: Gradient function
    - d2x: Hessian function
    - d3x: Third-order derivative function (not used, kept for compatibility)
    - x0: Initial point
    - max_iterations: Maximum iterations
    - tol: Convergence tolerance (gradient norm)
    - use_line_search: Whether to use line search (default: True)
    - min_eigenvalue_tol: Minimum eigenvalue tolerance to ensure local minimum (default: 1e-6)

    Returns:
    - x_curr: Final point
    - converged: Boolean indicating convergence to local minimum (not saddle point)
    - k: Number of iterations
    """
    k = 0
    x_curr = x0.copy().reshape(-1, 1)
    converged = False

    try:
        while k < max_iterations:
            k += 1
            grad = dx(x_curr)
            grad_norm = LA.norm(grad)

            # Compute Hessian
            hessian = d2x(x_curr)

            # Check convergence: gradient norm AND positive definite Hessian (local minimum)
            if grad_norm <= tol:
                # Check if Hessian is positive definite (local minimum, not saddle point)
                eigenvalues = LA.eigvalsh(hessian)
                min_eigenvalue = np.min(eigenvalues)

                if min_eigenvalue >= min_eigenvalue_tol:
                    # Positive definite Hessian -> local minimum
                    converged = True
                    break
                else:
                    # Negative eigenvalue -> saddle point, continue optimization
                    # Use negative eigenvector direction to escape saddle point
                    eigenvals, eigenvecs = LA.eigh(hessian)
                    neg_eigenvec_idx = np.where(eigenvals < -min_eigenvalue_tol)[0]

                    if len(neg_eigenvec_idx) > 0:
                        # Use most negative eigenvector as escape direction
                        escape_dir = eigenvecs[:, neg_eigenvec_idx[0]].reshape(-1, 1)
                        # Small step in negative eigenvector direction
                        escape_step = 0.1 * escape_dir
                        x_curr = x_curr + escape_step
                        continue

            # Compute Newton direction
            # Try to solve Newton system: H * p = -g
            try:
                # Try Cholesky decomposition (for positive definite Hessian)
                L = LA.cholesky(hessian)
                p = -LA.solve(L @ L.T, grad)
            except LA.LinAlgError:
                # If not positive definite, check eigenvalues
                eigenvalues = LA.eigvalsh(hessian)
                min_eigenvalue = np.min(eigenvalues)

                if min_eigenvalue < -min_eigenvalue_tol:
                    # Negative eigenvalue -> saddle point, use negative eigenvector
                    eigenvals, eigenvecs = LA.eigh(hessian)
                    neg_eigenvec_idx = np.where(eigenvals < -min_eigenvalue_tol)[0]
                    if len(neg_eigenvec_idx) > 0:
                        escape_dir = eigenvecs[:, neg_eigenvec_idx[0]].reshape(-1, 1)
                        # Combine with gradient descent direction
                        p = -grad + 0.5 * escape_dir
                    else:
                        p = -grad  # Fallback to gradient descent
                else:
                    # Near zero eigenvalue, use regularized Hessian
                    try:
                        reg_hess = hessian + max(
                            1e-8, abs(min_eigenvalue) + 1e-8
                        ) * np.eye(hessian.shape[0])
                        L = LA.cholesky(reg_hess)
                        p = -LA.solve(L @ L.T, grad)
                    except LA.LinAlgError:
                        # Use pseudo-inverse as last resort
                        p = -LA.lstsq(hessian, grad, rcond=None)[0]
                        p = p.reshape(-1, 1)

            # Determine step size
            if use_line_search:
                # Use backtracking line search
                alpha = backtracking_line_search(fx, dx, x_curr, p, alpha_init=1.0)
            else:
                # Use full Newton step
                alpha = 1.0

            # Update
            x_curr = x_curr + alpha * p

            # Check for stagnation
            if LA.norm(alpha * p) < tol * 1e-3:
                # Final check: ensure we're at a local minimum, not saddle point
                final_hessian = d2x(x_curr)
                final_eigenvalues = LA.eigvalsh(final_hessian)
                min_eigenvalue = np.min(final_eigenvalues)

                if min_eigenvalue >= min_eigenvalue_tol:
                    converged = True
                    break

        # Final convergence check
        if not converged:
            final_grad_norm = LA.norm(dx(x_curr))
            if final_grad_norm <= tol:
                # Check if it's a local minimum
                final_hessian = d2x(x_curr)
                final_eigenvalues = LA.eigvalsh(final_hessian)
                min_eigenvalue = np.min(final_eigenvalues)
                converged = min_eigenvalue >= min_eigenvalue_tol

    except Exception as e:
        # If any error occurs, mark as not converged
        converged = False

    return x_curr, converged, k


def unregularized_third_newton_run(
    fx, dx, d2x, d3x, x0, max_iterations, tol
): 
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
            # Compute dynamic SDP tolerance based on gradient norm (Inexact Newton)
            grad_norm = LA.norm(Dx)
            sdp_tol = compute_dynamic_sdp_tol(grad_norm, tol)
            out = min3(D3x, D2x, Dx, sdp_tol=sdp_tol)

            if "infeasible" in out[2] or "unbounded" in out[2]:
                alpha = alpha_approx(Dx, D2x, D3x)
                out_regularized = min3(D3x, D2x + alpha * np.eye(2), Dx, sdp_tol=sdp_tol)
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
