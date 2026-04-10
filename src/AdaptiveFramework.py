import numpy as np
import cvxpy as cvx
import numpy.linalg as LA

# ========================================================================================================================
# Adaptive Framework
# ========================================================================================================================

# Maximum threshold for sigma to prevent numerical overflow and infinite loops
SIGMA_MAX_THRESHOLD = 1e8

# Check solver availability once at module import time
_MOSEK_AVAILABLE = False
_CVXOPT_AVAILABLE = False


def compute_dynamic_sdp_tol(grad_norm, tol, sdp_tol_min=1e-6, sdp_tol_max=1e-3):
    """
    Compute dynamic SDP tolerance based on gradient norm (Inexact Newton approach).

    This function implements adaptive precision for SDP subproblems:
    - Early iterations (large gradient): use lower precision (faster)
    - Late iterations (small gradient): use higher precision (more accurate)

    Parameters:
    - grad_norm: Current gradient norm ||∇f(x)||
    - tol: Convergence tolerance for the outer algorithm
    - sdp_tol_min: Minimum SDP tolerance (used near convergence, default: 1e-6)
    - sdp_tol_max: Maximum SDP tolerance (used in early iterations, default: 1e-3)

    Returns:
    - sdp_tol: Dynamic SDP tolerance
    """
    # Normalize gradient norm relative to convergence tolerance
    if grad_norm <= tol:
        # Already converged, use highest precision
        return sdp_tol_min

    # Compute ratio: how far from convergence
    ratio = grad_norm / tol

    # Adaptive tolerance strategy:
    # - If ratio > 100 (far from convergence): use low precision (1e-3)
    # - If ratio < 10 (near convergence): use high precision (1e-6)
    # - Linear interpolation in between
    if ratio >= 100:
        return sdp_tol_max
    elif ratio <= 10:
        return sdp_tol_min
    else:
        # Linear interpolation on log scale
        log_ratio = np.log10(ratio)
        log_tol_min = np.log10(sdp_tol_min)
        log_tol_max = np.log10(sdp_tol_max)
        # Map [log10(10), log10(100)] to [log_tol_min, log_tol_max]
        log_tol = log_tol_min + (log_tol_max - log_tol_min) * (log_ratio - 1.0) / (
            2.0 - 1.0
        )
        return 10.0**log_tol


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


def almton_simple(
    fx,
    dx,
    d2x,
    d3x,
    x0,
    max_iterations,
    tol,
    param_list,
    verbose=True,
    sdp_solver="auto",
    sdp_tol=None,
):
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
    - verbose: Print iteration details (default True)
    - sdp_solver: SDP solver to use ('auto', 'mosek', 'scs', 'cvxopt')
                  'auto' (default): tries MOSEK if available, falls back to SCS
                  'mosek': use MOSEK (requires license)
                  'scs': use SCS (open source)
                  'cvxopt': use CVXOPT (open source)
    - sdp_tol: Fixed SDP tolerance (default: None, uses dynamic tolerance)
               If provided, overrides dynamic tolerance calculation

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
        - 'sdp_solves': Total number of SDP solves
        - 'rho_history': List of rho_k values
        - 'n_successful': Number of successful iterations
        - 'n_unsuccessful': Number of unsuccessful iterations
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
    sigma_exceeded = False  # Flag to track if sigma exceeded threshold

    # SDP counter
    sdp_counter = {"count": 0}

    # Initialize history lists
    x_history = [x0.copy()]
    f_history = [fx(x0)]
    s_history = []
    sigma_history = [sigma_k]
    success_history = []
    grad_norm_history = [LA.norm(dx(x0))]
    sigma_approx_history = []
    rho_history = []

    while k < max_iterations:
        # Step 1: Test for termination
        grad_k = dx(x_k)  # Shape (n, 1)
        grad_norm = LA.norm(grad_k)
        if grad_norm <= tol:
            if verbose:
                print(f"Converged at iteration {k} with ||∇f(x)|| = {grad_norm}")
            break

        # Step 2: Step calculation
        sigma_k = sigma_history[k]
        # Compute SDP tolerance: use fixed value if provided, otherwise use dynamic tolerance
        if sdp_tol is None:
            current_sdp_tol = compute_dynamic_sdp_tol(grad_norm, tol)
        else:
            current_sdp_tol = sdp_tol
        s_k, has_local_min = compute_step(
            x_k,
            sigma_k,
            dx,
            d2x,
            d3x,
            n,
            sdp_counter,
            solver=sdp_solver,
            sdp_tol=current_sdp_tol,
        )
        sigma_approx = alpha_approx(dx(x_k), d2x(x_k), d3x(x_k))

        if verbose:
            print("=" * 100)
            print(f"Iteration {k}:")
            print(f"  sigma LM: {sigma_approx[0][0]}")

        if has_local_min:
            bar_x = x_k + s_k  # Trial point
            lambda_k = compute_lambda_min(bar_x, sigma_k, d2x)
            if lambda_k < c:
                sigma_k = sigma_k + c - lambda_k
                # Use same SDP tolerance for consistency
                if sdp_tol is None:
                    current_sdp_tol = compute_dynamic_sdp_tol(grad_norm, tol)
                else:
                    current_sdp_tol = sdp_tol
                s_k, has_local_min = compute_step(
                    x_k,
                    sigma_k,
                    dx,
                    d2x,
                    d3x,
                    n,
                    sdp_counter,
                    solver=sdp_solver,
                    sdp_tol=current_sdp_tol,
                )
                if has_local_min:
                    bar_x = x_k + s_k
                else:
                    s_k = np.zeros_like(x_k)
        else:
            s_k = np.zeros_like(x_k)
            bar_x = x_k

        # Step 3: Acceptance of the trial point
        success = False
        rho_k = -np.inf
        if LA.norm(s_k) > 0:
            rho_k = compute_rho(fx, dx, d2x, d3x, x_k, bar_x, s_k, sigma_k)
            if rho_k >= eta:
                x_k = bar_x  # Accept the trial point
                success = True
        else:
            success = "pre-rejected"  # No step taken

        rho_history.append(rho_k)

        # Compute function value once
        f_val = fx(x_k)

        s_history.append(s_k.copy())
        success_history.append(success)
        x_history.append(x_k.copy())
        f_history.append(f_val)
        sigma_approx_history.append(sigma_approx)
        grad_norm_history.append(grad_norm)

        # Print iteration information
        if verbose:
            print(f"  x_k: {x_k.flatten()}")
            print(f"  s_k: {s_k.flatten()}")
            print(f"  sigma_k: {sigma_k}")
            print(f"  f(x_k): {f_val}")
            print(f"  Success: {success}")
            print(f"  rho_k: {rho_k}")
            print(f"  ||∇f(x_k)||: {grad_norm}")

        # Step 4: Regularization parameter update
        if success == True:
            sigma_k = 0.0  # Reset for successful iteration
        else:
            if sigma_k == 0.0:
                if verbose:
                    print("Warning: Regularization parameter is zero")
                sigma_k = 1.0
                if verbose:
                    print(f"  New sigma_k: {sigma_k}")
            else:
                sigma_k = gamma * sigma_k  # Increase sigma_k

        # Check for sigma explosion
        if sigma_k > SIGMA_MAX_THRESHOLD:
            sigma_exceeded = True
            if verbose:
                print(
                    f"\n[ALMTON ERROR] Sigma has exceeded the maximum threshold ({SIGMA_MAX_THRESHOLD:.1e})."
                )
                print(f"               Current sigma: {sigma_k}")
                print(
                    "               The algorithm is unable to find a solution at this point."
                )
                print("               Terminating optimization.")
            break

        # Append to history lists
        sigma_history.append(sigma_k)

        k += 1

    if k >= max_iterations and verbose:
        print(f"Maximum iterations {max_iterations} reached without convergence")

    # Compute statistics
    n_successful = sum(1 for s in success_history if s == True)
    n_unsuccessful = len(success_history) - n_successful

    # Determine if algorithm truly converged
    # True convergence requires: gradient norm <= tol AND no sigma explosion
    final_grad_norm = LA.norm(dx(x_k))
    truly_converged = (
        (final_grad_norm <= tol) and (not sigma_exceeded) and (k < max_iterations)
    )

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
        "rho_history": rho_history,
        "sdp_solves": sdp_counter["count"],
        "n_successful": n_successful,
        "n_unsuccessful": n_unsuccessful,
        "converged": truly_converged,
        "sigma_exceeded": sigma_exceeded,
    }


def compute_step(
    x_k, sigma_k, dx, d2x, d3x, n, sdp_counter=None, solver="auto", sdp_tol=1e-5
):
    """
    Compute the step s_k by solving the SDP for the local minimum of m_{f,x_k}(x, σ_k).

    Parameters:
    - x_k: Current point (n, 1)
    - sigma_k: Regularization parameter (float)
    - dx: Gradient function
    - d2x: Hessian function
    - d3x: Third-order derivative function
    - n: Dimension of the problem
    - sdp_counter: Dict to track SDP solves (optional)
    - solver: SDP solver to use ('auto', 'mosek', 'scs', 'cvxopt')
              'auto' tries MOSEK first if available, then falls back to SCS
    - sdp_tol: Tolerance for SDP solver (default: 1e-5)
               For inexact Newton: use lower tolerance (e.g., 1e-3) in early iterations,
               higher tolerance (e.g., 1e-6) near convergence

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

    # Use pre-checked solver availability (checked at module import time)
    # Determine solver list
    if solver == "auto":
        # Smart selection: prioritize MOSEK, fall back to SCS
        solver_list = []
        if _MOSEK_AVAILABLE:
            solver_list.append(("mosek", cvx.MOSEK, {}))
        solver_list.append(
            (
                "scs",
                cvx.SCS,
                {"eps": sdp_tol, "max_iters": 10000, "scale": 10.0, "normalize": True},
            )
        )
        # Add CVXOPT as a fallback if available
        if _CVXOPT_AVAILABLE:
            solver_list.append(("cvxopt", cvx.CVXOPT, {}))
    elif solver == "mosek":
        if not _MOSEK_AVAILABLE:
            print("Warning: MOSEK requested but not available, falling back to SCS")
            solver_list = [
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
            ]
        else:
            solver_list = [("mosek", cvx.MOSEK, {})]
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
            print("Warning: CVXOPT requested but not available, falling back to SCS")
            solver_list = [
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
            ]
    else:
        # Default to SCS
        solver_list = [
            (
                "scs",
                cvx.SCS,
                {"eps": sdp_tol, "max_iters": 10000, "scale": 10.0, "normalize": True},
            )
        ]

    # Try each solver in the solver list
    failure_reasons = []

    for solver_name, solver_obj, solver_params in solver_list:
        try:
            prob.solve(solver=solver_obj, verbose=False, **solver_params)
            if sdp_counter is not None:
                sdp_counter["count"] += 1
                if "solver_used" not in sdp_counter:
                    sdp_counter["solver_used"] = {}
                sdp_counter["solver_used"][solver_name] = (
                    sdp_counter["solver_used"].get(solver_name, 0) + 1
                )

            if prob.status in ["optimal", "optimal_inaccurate"]:
                s_k = y.value
                if s_k is None:
                    failure_reasons.append(
                        f"{solver_name} returned None (status: {prob.status})"
                    )
                    continue  # Try next solver
                return s_k, True
            else:
                failure_reasons.append(f"{solver_name} status: {prob.status}")
        except Exception as e:
            # Current solver failed, try next one
            failure_reasons.append(f"{solver_name} exception: {str(e)}")
            continue

    # All solvers failed
    print(f"\n[SDP ERROR] Failed to solve SDP at x = {x_k.flatten()}")
    print(f"            Sigma: {sigma_k}")
    print(f"            Solvers tried: {', '.join(failure_reasons)}")
    print(
        "            This point appears to be numerically intractable for the available solvers."
    )

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


def compute_model_value(fx, dx, d2x, d3x, x_k, bar_x, sigma_tilde):
    """
    Compute the model value m_{f,x_k}(bar_x; sigma_tilde).

    Parameters:
    - fx: Objective function
    - dx: Gradient function
    - d2x: Hessian function
    - d3x: Third-order derivative function
    - x_k: Current point (n, 1)
    - bar_x: Trial point (n, 1)
    - sigma_tilde: Regularization parameter (float)

    Returns:
    - m_value: Model value (float)
    """
    s_k = bar_x - x_k
    # Third-order Taylor approximation
    phi_value = compute_phi(fx, dx, d2x, d3x, x_k, s_k)
    # Add regularization term
    m_value = phi_value + sigma_tilde * (LA.norm(s_k) ** 2)
    return float(m_value)


# ========================================================================================================================
# New Adaptive Framework
# ========================================================================================================================
def almton_heuristic(
    fx,
    dx,
    d2x,
    d3x,
    x0,
    max_iterations,
    tol,
    param_list,
    verbose=True,
    sdp_solver="auto",
    sdp_tol=None,
):
    """
    Adaptive Levenberg-Marquardt Third-Order Newton's Method (ALMTON) - Version 2.
    This version implements the updated algorithm with improved regularization
    parameter update strategy and model phase with multiple checks.
    """
    # Unpack parameters
    c, l, eta, gamma = param_list
    if not (c > 0 and l > 0 and 0 < l <= c / 6 and 0 < eta < 1 and gamma > 1):
        raise ValueError(
            "Parameters must satisfy: c > 0, l ∈ (0, c/6], eta ∈ (0, 1), gamma > 1"
        )

    # Initialize variables
    x_k = x0.copy()  # Shape (n, 1)
    k = 0  # Iteration counter
    n = x0.shape[0]  # Dimension of the problem
    sigma_k = 0.0
    sigma_exceeded = False  # Flag to track if sigma exceeded threshold

    # SDP counter
    sdp_counter = {"count": 0}

    # Initialize history lists
    x_history = [x0.copy()]
    f_history = [fx(x0)]
    s_history = []
    sigma_history = [sigma_k]
    sigma_tilde_history = []
    success_history = []
    grad_norm_history = [LA.norm(dx(x0))]
    rho_history = []

    while k < max_iterations:
        # Step 1: Test for termination
        grad_k = dx(x_k)  # Shape (n, 1)
        grad_norm = LA.norm(grad_k)

        if grad_norm <= tol:
            if verbose:
                print(f"Converged at iteration {k}, ||∇f(x)|| = {grad_norm}")
            break

        # Compute alpha_LM threshold
        alpha_LM = alpha_approx(grad_k, d2x(x_k), d3x(x_k))
        f_xk = fx(x_k)

        if verbose:
            print("=" * 100)
            print(f"Iteration {k}:")
            print(f"  x_k: {x_k.flatten()}")
            print(f"  f(x_k): {f_xk}")
            print(f"  ||∇f(x_k)||: {grad_norm}")
            print(
                f"  alpha_LM: {alpha_LM if isinstance(alpha_LM, float) else alpha_LM[0][0]}"
            )

        # Step 2: Step calculation (model phase)
        sigma_tilde = sigma_k
        max_inner_iterations = 100  # Safety limit for the repeat loop
        inner_iter = 0

        while inner_iter < max_inner_iterations:
            # Compute SDP tolerance: use fixed value if provided, otherwise use dynamic tolerance
            if sdp_tol is None:
                current_sdp_tol = compute_dynamic_sdp_tol(grad_norm, tol)
            else:
                current_sdp_tol = sdp_tol
            # Try to find a local minimizer with current sigma_tilde
            s_k, has_local_min = compute_step(
                x_k,
                sigma_tilde,
                dx,
                d2x,
                d3x,
                n,
                sdp_counter,
                solver=sdp_solver,
                sdp_tol=current_sdp_tol,
            )

            if has_local_min and LA.norm(s_k) > 0:
                bar_x = x_k + s_k  # Trial point

                # Compute bar_lambda_k
                bar_lambda_k = compute_lambda_min(bar_x, 2 * sigma_tilde, d2x)

                # Compute model value
                m_value = compute_model_value(fx, dx, d2x, d3x, x_k, bar_x, sigma_tilde)

                # Check all conditions
                condition_lambda = bar_lambda_k >= c
                condition_model = m_value <= f_xk

                if condition_lambda and condition_model:
                    # All invariants satisfied, break
                    if verbose:
                        print(
                            f"  [Inner iteration {inner_iter}] Found valid step, sigma_tilde = {sigma_tilde}"
                        )
                    break
                else:
                    # Need to increase sigma_tilde
                    if verbose:
                        if not condition_lambda:
                            print(
                                f"  [Inner iteration {inner_iter}] bar_lambda_k = {bar_lambda_k} < c = {c}"
                            )
                        if not condition_model:
                            print(
                                f"  [Inner iteration {inner_iter}] m_value = {m_value} > f(x_k) = {f_xk}"
                            )

                    # Compute new sigma_tilde
                    alpha_val = (
                        alpha_LM
                        if isinstance(alpha_LM, float)
                        else float(alpha_LM[0][0])
                    )
                    sigma_tilde = max(
                        alpha_val,
                        gamma * max(1, sigma_tilde),
                        sigma_tilde + max(0, c - bar_lambda_k),
                    )
                    if verbose:
                        print(
                            f"  [Inner iteration {inner_iter}] Increased sigma_tilde to {sigma_tilde}"
                        )

                    if sigma_tilde > SIGMA_MAX_THRESHOLD:
                        sigma_exceeded = True
                        if verbose:
                            print(
                                f"\n[almton_heuristic ERROR] Sigma_tilde exceeded threshold ({SIGMA_MAX_THRESHOLD:.1e}) during inner loop."
                            )
                        break
            else:
                # No local minimizer at current sigma_tilde
                if verbose:
                    print(f"  [Inner iteration {inner_iter}] No local minimizer found")
                alpha_val = (
                    alpha_LM if isinstance(alpha_LM, float) else float(alpha_LM[0][0])
                )
                sigma_tilde = max(alpha_val, gamma * max(1, sigma_tilde))
                if verbose:
                    print(
                        f"  [Inner iteration {inner_iter}] Increased sigma_tilde to {sigma_tilde}"
                    )

                if sigma_tilde > SIGMA_MAX_THRESHOLD:
                    sigma_exceeded = True
                    if verbose:
                        print(
                            f"\n[almton_heuristic ERROR] Sigma_tilde exceeded threshold ({SIGMA_MAX_THRESHOLD:.1e}) during inner loop."
                        )
                    break

            inner_iter += 1

        if sigma_tilde > SIGMA_MAX_THRESHOLD:
            if verbose:
                print(
                    "               The algorithm is unable to find a solution at this point."
                )
                print("               Terminating optimization.")
            break

        if inner_iter >= max_inner_iterations:
            if verbose:
                print(
                    f"Warning: Inner loop reached maximum iterations {max_inner_iterations}"
                )
            s_k = np.zeros_like(x_k)
            bar_x = x_k

        # Record phase-closing sigma_tilde
        sigma_tilde_k = sigma_tilde
        sigma_tilde_history.append(sigma_tilde_k)

        if verbose:
            print(f"  s_k: {s_k.flatten()}")
            print(f"  sigma_tilde_k (phase-closing value): {sigma_tilde_k}")

        # Step 3: Acceptance of the trial point
        success = False
        rho_k = -np.inf

        if LA.norm(s_k) > 0:
            f_bar_x = fx(bar_x)
            numerator = f_xk - f_bar_x

            # Compute rho_k according to new formula
            if sigma_tilde_k == 0:
                denominator = l * (LA.norm(s_k) ** 2)
            else:
                m_bar_x = compute_model_value(
                    fx, dx, d2x, d3x, x_k, bar_x, sigma_tilde_k
                )
                denominator = f_xk - m_bar_x

            if denominator > 0:
                rho_k = numerator / denominator
            else:
                rho_k = -np.inf

            if rho_k >= eta:
                x_k = bar_x  # Accept the trial point
                success = True
                if verbose:
                    print(f"  Trial point accepted: rho_k = {rho_k} >= eta = {eta}")
            else:
                if verbose:
                    print(f"  Trial point rejected: rho_k = {rho_k} < eta = {eta}")
        else:
            if verbose:
                print(f"  Zero step, rejected")
            success = False

        # Update histories
        s_history.append(s_k.copy())
        success_history.append(success)
        x_history.append(x_k.copy())
        f_history.append(fx(x_k))
        grad_norm_history.append(LA.norm(dx(x_k)))
        rho_history.append(rho_k)

        # Step 4: Regularization parameter update
        if rho_k >= eta:
            sigma_k = 0.0
            if verbose:
                print(f"  Successful iteration, reset sigma_{k+1} = 0")
        else:
            if sigma_tilde_k == 0:
                alpha_val = (
                    alpha_LM if isinstance(alpha_LM, float) else float(alpha_LM[0][0])
                )
                sigma_k = max(alpha_val, gamma)
                if verbose:
                    print(
                        f"  Failed iteration (sigma_tilde=0), set sigma_{k+1} = max(alpha_LM, gamma) = {sigma_k}"
                    )
            else:
                sigma_k = gamma * sigma_tilde_k
                if verbose:
                    print(
                        f"  Failed iteration (sigma_tilde>0), set sigma_{k+1} = gamma * sigma_tilde = {sigma_k}"
                    )

        # Check for sigma explosion
        if sigma_k > SIGMA_MAX_THRESHOLD:
            sigma_exceeded = True
            if verbose:
                print(
                    f"\n[almton_heuristic ERROR] Sigma has exceeded the maximum threshold ({SIGMA_MAX_THRESHOLD:.1e})."
                )
                print(f"               Current sigma: {sigma_k}")
                print(
                    "               The algorithm is unable to find a solution at this point."
                )
                print("               Terminating optimization.")
            break

        # Append to history lists
        sigma_history.append(sigma_k)

        k += 1

    if k >= max_iterations and verbose:
        print(f"Reached maximum iterations {max_iterations}, did not converge")

    # Compute statistics
    n_successful = sum(1 for s in success_history if s == True)
    n_unsuccessful = len(success_history) - n_successful

    # Determine if algorithm truly converged
    # True convergence requires: gradient norm <= tol AND no sigma explosion
    final_grad_norm = LA.norm(dx(x_k))
    truly_converged = (
        (final_grad_norm <= tol) and (not sigma_exceeded) and (k < max_iterations)
    )

    return {
        "x_final": x_k,
        "iterations": k,
        "x_history": x_history,
        "s_history": s_history,
        "sigma_history": sigma_history,
        "sigma_tilde_history": sigma_tilde_history,
        "f_history": f_history,
        "success_history": success_history,
        "grad_norm_history": grad_norm_history,
        "rho_history": rho_history,
        "sdp_solves": sdp_counter["count"],
        "n_successful": n_successful,
        "n_unsuccessful": n_unsuccessful,
        "converged": truly_converged,
        "sigma_exceeded": sigma_exceeded,
    }


# ========================================================================================================================
# ALMTON with Interpolation (ALMTON-Interp)
# ========================================================================================================================

# Maximum threshold for sigma (more lenient than ALMTON/almton_heuristic's 1e8)
SIGMA_MAX_THRESHOLD_INTERP = 1e10


# ============================================================================
# Helper Functions for Polynomials
# ============================================================================


def padded_polyder(poly):
    """Compute polynomial derivative with padding to keep size consistent."""
    der = np.polyder(poly)
    return der


def real_roots(poly):
    """Find real roots of a polynomial."""
    if len(poly) == 0:
        return np.array([])
    # Remove leading zeros to avoid numpy errors
    while len(poly) > 1 and np.isclose(poly[0], 0):
        poly = poly[1:]

    roots_all = np.roots(poly)
    # Filter real roots (imaginary part is negligible)
    real_roots_arr = roots_all[np.abs(roots_all.imag) < 1e-10].real
    return real_roots_arr


def construct_taylor_poly_almton(s, f, g, H, T):
    """
    Construct 1D Taylor polynomial t(alpha) along step direction s.
    t(alpha) = c3*alpha^3 + c2*alpha^2 + c1*alpha + c0
    """
    s = s.flatten()
    g = g.flatten()

    T_s = np.tensordot(T, s, axes=([0], [0]))  # Shape (n, n)
    c3 = (1.0 / 6.0) * s @ T_s @ s
    c2 = 0.5 * s @ H @ s
    c1 = g @ s
    c0 = f

    return np.array([c3, c2, c1, c0])


# ============================================================================
# ALMTON Specific Logic (Pre-rejection & Interpolation)
# ============================================================================


def analyze_persistent_min_almton(taylor_poly):
    """
    Analyze persistent minimizers for ALMTON (LM Regularization).

    Unlike AR3 (4th order reg), ALMTON uses 2nd order reg.
    The critical boundary for persistent minimizers is derived from:
        alpha * t''(alpha) - t'(alpha) = 0
    Which simplifies to:
        3 * c3 * alpha^2 - c1 = 0

    Parameters:
    - taylor_poly: [c3, c2, c1, c0]

    Returns:
    - persistent_alpha_limit: The boundary alpha.
      If step_size > this limit, the step is likely transient.
      Returns np.inf if always persistent.
    """
    c3, c2, c1, _ = taylor_poly

    if np.abs(c3) < 1e-10:
        return np.inf

    val = c1 / (3.0 * c3)

    if val > 0:
        return np.sqrt(val)
    else:
        return np.inf


def find_sigma_for_decrease_almton(taylor_poly, norm_s, target_decrease):
    """
    Find sigma such that the model decrease matches target_decrease.

    Derived polynomial for alpha:
        0.5 c3 * alpha^3 - 0.5 c1 * alpha - target = 0
    """
    c3, c2, c1, c0 = taylor_poly

    poly_eqn = np.array(
        [
            0.5 * c3,
            0.0,  # alpha^2 term cancels out exactly for LM regularization!
            -0.5 * c1,
            -target_decrease,
        ]
    )

    roots = real_roots(poly_eqn)
    valid_alphas = roots[roots > 1e-6]

    if len(valid_alphas) == 0:
        return None

    best_sigma = None

    for alpha in valid_alphas:
        t_prime = 3 * c3 * (alpha**2) + 2 * c2 * alpha + c1
        sigma_val = -t_prime / (2.0 * alpha * (norm_s**2))

        if sigma_val > 0:
            if best_sigma is None:
                best_sigma = sigma_val
            else:
                best_sigma = sigma_val

    return best_sigma


# ============================================================================
# Main ALMTON-Interp Algorithm
# ============================================================================


def almton_interp(
    fx,
    dx,
    d2x,
    d3x,
    x0,
    max_iterations,
    tol,
    param_list=None,
    verbose=True,
    sdp_solver="auto",
):
    """
    ALMTON with Interpolation Update and Pre-rejection.

    Integrates the logic from AR3-Interp into the ALMTON (LM-regularized) framework.

    Parameters:
    - param_list: [eta1, eta2, gamma, sigma_min, beta]
        eta1: Unsuccessful cutoff (e.g. 0.01)
        eta2: Very successful cutoff (e.g. 0.9)
        gamma: Safety factor for increasing sigma (e.g. 2.0)
        sigma_min: Minimum regularization (e.g. 1e-8)
        beta: Interpolation fraction for very successful steps (e.g. 0.1)
    """

    # Default parameters if not provided
    if param_list is None:
        eta1, eta2, gamma, sigma_min, beta = 0.01, 0.95, 2.0, 1e-8, 0.1
    else:
        eta1 = param_list[0]
        eta2 = param_list[1]
        gamma = param_list[2] if len(param_list) > 2 else 2.0
        sigma_min = param_list[3] if len(param_list) > 3 else 1e-8
        beta = param_list[4] if len(param_list) > 4 else 0.1

    # Initialize
    x_k = x0.copy()
    n = x0.shape[0]
    k = 0
    sigma_exceeded = False

    g0 = dx(x0)
    sigma_k = max(1.0, sigma_min)

    # Histories
    x_history = [x0.copy()]
    f_history = [fx(x0)]
    grad_norm_history = [LA.norm(g0)]
    sigma_history = [sigma_k]
    rho_history = []
    success_history = []

    # Counters
    counters = {
        "sdp_solves": 0,
        "pre_rejections": 0,
        "interp_updates": 0,
        "successful": 0,
    }

    sdp_tracker = {"count": 0}

    while k < max_iterations:
        # 1. Derivatives
        g_k = dx(x_k)
        f_k = fx(x_k)
        H_k = d2x(x_k)
        T_k = d3x(x_k)

        grad_norm = LA.norm(g_k)

        # 2. Termination
        if grad_norm <= tol:
            if verbose:
                print(f"Converged at iter {k}, grad_norm={grad_norm:.2e}")
            break

        # 3. Compute Step (Model Phase) using existing ALMTON SDP logic
        s_k = np.zeros((n, 1))
        model_phase_done = False

        while not model_phase_done:
            sdp_tol = compute_dynamic_sdp_tol(grad_norm, tol)
            s_k, has_local_min = compute_step(
                x_k,
                sigma_k,
                dx,
                d2x,
                d3x,
                n,
                sdp_tracker,
                solver=sdp_solver,
                sdp_tol=sdp_tol,
            )

            if has_local_min and LA.norm(s_k) > 1e-14:
                bar_x = x_k + s_k
                lambda_min = compute_lambda_min(bar_x, 2 * sigma_k, d2x)

                if lambda_min < 1e-4:
                    sigma_k = max(gamma * sigma_k, sigma_k + 1e-4 - lambda_min)
                    if verbose:
                        print(f"  Increasing sigma (curvature): {sigma_k:.2e}")
                else:
                    model_phase_done = True
            else:
                sigma_k = max(gamma * sigma_k, 1.0)
                if verbose:
                    print(f"  Increasing sigma (no local min): {sigma_k:.2e}")

            if sigma_k > SIGMA_MAX_THRESHOLD_INTERP:
                sigma_exceeded = True
                if verbose:
                    print(
                        f"\n[ALMTON-Interp ERROR] Sigma has exceeded the maximum threshold ({SIGMA_MAX_THRESHOLD_INTERP:.1e})."
                    )
                    print(f"               Current sigma: {sigma_k}")
                    print(
                        "               The algorithm is unable to find a solution at this point."
                    )
                break

        norm_s = LA.norm(s_k)
        counters["sdp_solves"] = sdp_tracker["count"]

        # 4. Pre-rejection Framework
        taylor_poly = construct_taylor_poly_almton(s_k, f_k, g_k, H_k, T_k)
        persistent_limit = analyze_persistent_min_almton(taylor_poly)
        is_transient = 1.0 > (persistent_limit + 1e-6)

        if is_transient:
            if verbose:
                print(
                    f"  Iter {k}: Pre-rejected (Transient). Limit={persistent_limit:.2f}"
                )
            rho_k = -1.0
            actual_decrease = 0.0
            counters["pre_rejections"] += 1
            step_accepted = False
        else:
            # 5. Evaluation
            f_new = fx(x_k + s_k)
            actual_decrease = f_k - f_new

            t_val = np.polyval(taylor_poly, 1.0)
            m_val = t_val + sigma_k * (norm_s**2)
            predicted_decrease = f_k - m_val

            if predicted_decrease < 1e-14:
                rho_k = -np.inf
            else:
                rho_k = actual_decrease / predicted_decrease

            step_accepted = rho_k >= eta1

        # 6. Update Strategy (Interpolation)
        next_sigma = sigma_k

        if step_accepted:
            x_k = x_k + s_k
            x_history.append(x_k.copy())
            counters["successful"] += 1

            if rho_k >= eta2:
                sigma_interp = find_sigma_for_decrease_almton(
                    taylor_poly, norm_s, actual_decrease
                )

                if sigma_interp is not None and sigma_interp < sigma_k:
                    next_sigma = sigma_interp
                    counters["interp_updates"] += 1
                else:
                    next_sigma = max(sigma_min, sigma_k / gamma)

                next_sigma = min(next_sigma, 1.0)

            else:
                next_sigma = max(sigma_min, sigma_k * 0.5)
        else:
            x_history.append(x_k.copy())
            next_sigma_base = sigma_k * gamma

            if not is_transient and actual_decrease > 0:
                target_pred = actual_decrease / eta1
                sigma_interp = find_sigma_for_decrease_almton(
                    taylor_poly, norm_s, target_pred
                )

                if sigma_interp is not None:
                    next_sigma = max(next_sigma_base, sigma_interp)
                    counters["interp_updates"] += 1
                else:
                    next_sigma = next_sigma_base
            else:
                next_sigma = next_sigma_base

        # Update History
        f_history.append(fx(x_k))
        grad_norm_history.append(LA.norm(dx(x_k)))
        sigma_history.append(sigma_k)
        rho_history.append(rho_k)
        success_history.append(step_accepted)

        sigma_k = next_sigma

        if sigma_k > SIGMA_MAX_THRESHOLD_INTERP:
            sigma_exceeded = True
            if verbose:
                print(
                    f"\n[ALMTON-Interp ERROR] Sigma has exceeded the maximum threshold ({SIGMA_MAX_THRESHOLD_INTERP:.1e})."
                )
                print(f"               Current sigma: {sigma_k}")
                print(
                    "               The algorithm is unable to find a solution at this point."
                )
            break

        k += 1

        if verbose:
            print(
                f"  Iter {k-1}: rho={rho_k:.4f}, sigma={sigma_history[-1]:.2e} -> {sigma_k:.2e}, Success={step_accepted}"
            )

    final_grad_norm = LA.norm(dx(x_k))
    converged = (
        (final_grad_norm <= tol) and (not sigma_exceeded) and (k < max_iterations)
    )

    return {
        "x_final": x_k,
        "iterations": k,
        "converged": converged,
        "x_history": x_history,
        "f_history": f_history,
        "grad_norm_history": grad_norm_history,
        "sigma_history": sigma_history,
        "rho_history": rho_history,
        "success_history": success_history,
        "stats": counters,
        "sigma_exceeded": sigma_exceeded,
    }
