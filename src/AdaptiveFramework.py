import numpy as np
import cvxpy as cvx
import numpy.linalg as LA


def almton(fx, dx, d2x, d3x, x0, max_iterations, tol, param_list):
    """
    Adaptive Unregularized Third-Order Newton's Method with Levenberg-Marquardt Regularization.

    Parameters:
    - fx: Objective function f(x): R^n -> R
    - dx: Gradient function ∇f(x): R^n -> R^n
    - d2x: Hessian function ∇²f(x): R^n -> R^(n×n)
    - d3x: Third-order derivative function ∇³f(x): R^n -> R^(n×n×n)
    - x0: Initial point (numpy array of shape (n, 1))
    - max_iterations: Maximum number of iterations (int)
    - tol: Convergence tolerance for ||∇f(x)|| (float)
    - param_list: List of parameters [c, l, eta, gamma] where:
        - c: Small positive constant for eigenvalue threshold
        - l: Small positive constant (not used in this implementation but included for completeness)
        - eta: Threshold for accepting trial point
        - gamma: Growth factor for regularization parameter

    Returns:
    - dict: Dictionary containing:
        - 'x_final': Final point (numpy array of shape (n, 1))
        - 'iterations': Number of iterations performed (int)
        - 'x_history': List of x_k at each iteration (including initial point)
        - 's_history': List of steps s_k at each iteration
        - 'sigma_history': List of sigma_k at each iteration
        - 'f_history': List of f(x_k) at each iteration
        - 'success_history': List of success flags at each iteration
        - 'grad_norm_history': List of ||∇f(x_k)|| at each iteration
    """
    # Unpack parameters
    c, l, eta, gamma = param_list
    if not (c > 0 and l > 0 and eta > 0 and gamma > 1):
        raise ValueError("Parameters must satisfy: c > 0, l > 0, eta > 0, gamma > 1")

    # Initialize variables
    x_k = x0.copy()  # Shape (n, 1)
    k = 0  # Iteration counter
    n = x0.shape[0]  # Dimension of the problem
    sigma_k = 0.0

    # Initialize history lists
    x_history = [x0.copy()]
    f_history = [fx(x0)]
    s_history = []
    sigma_history = [sigma_k]
    success_history = []
    grad_norm_history = [dx(x0)]
    sigma_approx_history = []

    while k < max_iterations:
        # Step 1: Test for termination
        grad_k = dx(x_k)  # Shape (n, 1)
        if LA.norm(grad_k) <= tol:
            print(f"Converged at iteration {k} with ||∇f(x)|| = {LA.norm(grad_k)}")
            break
        grad_norm_history.append(grad_k)

        # Step 2: Step calculation
        sigma_k = sigma_history[k]
        s_k, has_local_min = compute_step(x_k, sigma_k, dx, d2x, d3x, n)
        sigma_approx = alpha_approx(dx(x_k), d2x(x_k), d3x(x_k))
        s_k_approx, has_local_min_approx = compute_step(
            x_k, sigma_approx, dx, d2x, d3x, n
        )
        print("=" * 100)
        print(f"Iteration {k}:")
        print(f"  sigma LM: {sigma_approx[0][0]}")
        print(f"  s_k LM: {s_k_approx.flatten()}")

        if has_local_min:
            bar_x = x_k + s_k  # Trial point
            lambda_k = compute_lambda_min(bar_x, sigma_k, d2x)
            if lambda_k < c:
                sigma_k = sigma_k + c - lambda_k
                s_k, has_local_min = compute_step(x_k, sigma_k, dx, d2x, d3x, n)
                if has_local_min:
                    bar_x = x_k + s_k
                else:
                    s_k = np.zeros_like(x_k)
        else:
            s_k = np.zeros_like(x_k)
            bar_x = x_k

        # # Verify model condition
        # if LA.norm(s_k) > 0:
        #     m_value = (
        #         compute_phi(fx, dx, d2x, d3x, x_k, s_k) + sigma_k * LA.norm(s_k) ** 2
        #     )
        #     # if m_value > fx(x_k):
        #     #     print(f"Warning: Model value not less than f(x_k) at iteration {k}")
        #     #     print('Value difference', m_value - fx(x_k))
        #     if fx(x_k) < compute_phi(fx, dx, d2x, d3x, x_k, s_k):
        #         print(f"Warning: Model value not less than f(x_k) at iteration {k}")
        #         print('Value difference', m_value - fx(x_k))

        # Step 3: Acceptance of the trial point
        success = False
        if LA.norm(s_k) > 0:
            rho_k = compute_rho(fx, dx, d2x, d3x, x_k, bar_x, s_k, sigma_k)
            if rho_k >= eta:
                x_k = bar_x  # Accept the trial point
                success = True
        else:
            success = "pre-rejected"  # No step taken
        # If s_k = 0 or rho_k < eta, x_k remains unchanged

        # Compute function value once
        f_val = fx(x_k)

        s_history.append(s_k.copy())
        success_history.append(success)
        x_history.append(x_k.copy())
        f_history.append(f_val)
        sigma_approx_history.append(sigma_approx)

        # Print iteration information
        print(f"  x_k: {x_k.flatten()}")
        print(f"  s_k: {s_k.flatten()}")
        print(f"  sigma_k: {sigma_k}")
        print(f"  f(x_k): {f_val}")
        print(f"  Success: {success}")
        print(f"  ||∇f(x_k)||: {LA.norm(grad_k)}")

        # Step 4: Regularization parameter update
        if success == True:
            sigma_k = 0.0  # Reset for successful iteration
        else:
            if sigma_k == 0.0:
                print("Warning: Regularization parameter is zero")
                sigma_k = 1.0
                print(f"  New sigma_k: {sigma_k}")
            else:
                sigma_k = gamma * sigma_k  # Increase sigma_k

        # Append to history lists
        sigma_history.append(sigma_k)

        k += 1

    if k >= max_iterations:
        print(f"Maximum iterations {max_iterations} reached without convergence")

    # delete all the variables to avoid memory leaks
    del n, sigma_k

    return {
        "x_final": x_k,
        "iterations": k,
        "x_history": x_history,
        "s_history": s_history,
        "sigma_history": sigma_history,
        "f_history": f_history,
        "success_history": success_history,
        "grad_norm_history": grad_norm_history,
        "sigma_approx_history": sigma_approx_history,
        "converged": k < max_iterations,
    }


def compute_step(x_k, sigma_k, dx, d2x, d3x, n):
    """
    Compute the step s_k by solving the SDP for the local minimum of m_{f,x_k}(x, σ_k).

    Parameters:
    - x_k: Current point (n, 1)
    - sigma_k: Regularization parameter (float)
    - dx: Gradient function
    - d2x: Hessian function
    - d3x: Third-order derivative function
    - n: Dimension of the problem

    Returns:
    - s_k: Step direction (n, 1)
    - has_local_min: Boolean indicating if a local minimum was found
    """
    # Get derivatives at x_k
    H = d3x(x_k)  # Shape (n, n, n)
    Q = d2x(x_k)  # Shape (n, n)
    b = dx(x_k)  # Shape (n, 1)

    # Adjust Hessian with Levenberg-Marquardt regularization
    Q_reg = Q + sigma_k * np.eye(n)

    # SDP formulation
    Y = cvx.Variable((n, n), symmetric=True)
    y = cvx.Variable((n, 1))
    z = cvx.Variable((1, 1))
    L = cvx.Variable((n, 1))

    constraints = []
    Hess = Q_reg + sum([H[i, :, :] * y[i] for i in range(n)])
    constraints += [
        L[i] == cvx.trace(H[i, :, :] @ Y) + Q_reg[i, :] @ y for i in range(n)
    ]
    constraints += [
        Q_reg[i, :] @ y + cvx.trace(H[i, :, :] @ Y) / 2 + b[i] == 0 for i in range(n)
    ]

    T = cvx.bmat([[Hess, L], [L.T, z]])
    YYT = cvx.bmat([[Y, y], [y.T, np.ones((1, 1))]])
    constraints += [T >> 0, YYT >> 0]

    objective = cvx.Minimize(cvx.trace(Q_reg @ Y) / 2 + b.T @ y + z / 2)
    prob = cvx.Problem(objective, constraints)

    try:
        prob.solve(solver=cvx.SCS, verbose=False)
        if prob.status in ["optimal", "optimal_inaccurate"]:
            s_k = y.value
            if s_k is None:
                return np.zeros((n, 1)), False
            return s_k, True
        else:
            return np.zeros((n, 1)), False
    except:
        return np.zeros((n, 1)), False


def compute_lambda_min(bar_x, sigma_k, d2x):
    """
    Compute the minimum eigenvalue of ∇²f(bar_x) + σ_k I.

    Parameters:
    - bar_x: Trial point (n, 1)
    - sigma_k: Regularization parameter (float)
    - d2x: Hessian function

    Returns:
    - lambda_min: Minimum eigenvalue (float)
    """
    hessian = d2x(bar_x)  # Shape (n, n)
    eigenvalues = LA.eigvalsh(hessian + sigma_k * np.eye(len(bar_x)))
    return np.min(eigenvalues)


def compute_rho(fx, dx, d2x, d3x, x_k, bar_x, s_k, sigma_k):
    """
    Compute the ratio ρ_k for accepting the trial point.

    Parameters:
    - fx: Objective function
    - dx: Gradient function
    - d2x: Hessian function
    - d3x: Third-order derivative function
    - x_k: Current point (n, 1)
    - bar_x: Trial point (n, 1)
    - s_k: Step (n, 1)
    - sigma_k: Regularization parameter (float)

    Returns:
    - rho_k: Acceptance ratio (float)
    """
    f_xk = fx(x_k)
    f_bar_x = fx(bar_x)
    numerator = f_xk - f_bar_x

    if sigma_k == 0:
        denominator = LA.norm(s_k) ** 2
    else:
        phi_bar_x = compute_phi(fx, dx, d2x, d3x, x_k, s_k)
        denominator = f_xk - phi_bar_x

    if denominator <= 0:
        return -np.inf  # Reject if denominator is non-positive
    return numerator / denominator


def compute_phi(fx, dx, d2x, d3x, x_k, s_k):
    """
    Compute the third-order Taylor approximation Φ_{f,x_k}^3(x_k + s_k).

    Parameters:
    - fx: Objective function
    - dx: Gradient function
    - d2x: Hessian function
    - d3x: Third-order derivative function
    - x_k: Current point (n, 1)
    - s_k: Step (n, 1)

    Returns:
    - phi: Taylor approximation value (float)
    """
    f_xk = fx(x_k)
    grad = dx(x_k)  # Shape (n, 1)
    hess = d2x(x_k)  # Shape (n, n)
    third = d3x(x_k)  # Shape (n, n, n)

    # First-order term: ∇f(x_k)^T s_k
    first = grad.T @ s_k

    # Second-order term: (1/2) s_k^T ∇²f(x_k) s_k
    second = 0.5 * (s_k.T @ hess @ s_k)

    # Third-order term: (1/6) ∑_i (s_k)_i (s_k^T ∇³f(x_k)_i s_k)
    third_term = 0
    for i in range(len(s_k)):
        third_term += (1 / 6) * s_k[i] * (s_k.T @ third[i, :, :] @ s_k)

    phi = f_xk + first + second + third_term
    return float(phi)


def alpha_approx(Dx, D2x, D3x):

    n_dx = LA.norm(Dx)
    n = len(Dx)
    ns_d3x = np.array([LA.norm(D3x[i, :, :]) for i in range(n)]).reshape((n, 1))
    n_d3x = LA.norm(ns_d3x, 2)

    e_val, e_vec = LA.eigh(D2x)
    idx = np.argsort(e_val)
    e_val = e_val[idx]
    e_vec = e_vec[:, idx]
    e_min = np.min(e_val[0], 0)

    return np.sqrt(1.5 * (n_dx * n_d3x + ns_d3x.T @ np.abs(Dx))) - e_min
