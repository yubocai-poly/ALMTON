"""
AR3 (Adaptive Regularization using Third-order derivatives) Algorithm

Python implementation of the AR3 algorithm for unconstrained optimization.
This implementation follows the interface design of almton_heuristic for consistency.

Includes:
- ar3: Basic AR3 with Simple Update
- ar3_interp: AR3 with Interpolation Update and Pre-rejection (AR3-Interp+)
"""

import numpy as np
import numpy.linalg as LA
from scipy.optimize import minimize

try:
    from AR2 import ar2_simple

    AR2_AVAILABLE = True
except ImportError:
    AR2_AVAILABLE = False


def ar3_model(s, f, g, H, T, sigma):
    """
    Compute the AR3 regularized cubic model value.

    Model: m(s) = f + g'*s + (1/2)*s'*H*s + (1/6)*s'*T*s + (1/4)*sigma*||s||^4

    Parameters:
    - s: Step vector (n, 1) or (n,)
    - f: Function value at current point (scalar)
    - g: Gradient at current point (n, 1) or (n,)
    - H: Hessian at current point (n, n)
    - T: Third-order derivative tensor (n, n, n)
    - sigma: Regularization parameter (scalar)

    Returns:
    - m: Model value (scalar)
    """
    s = s.flatten()
    g = g.flatten()

    # Third-order term: (1/6) * s' * T * s
    # where (T*s)_ij = sum_k T_ijk * s_k
    T_s = np.tensordot(T, s, axes=([0], [0]))  # Shape: (n, n)
    third_order_term = (1 / 6) * s @ T_s @ s

    # Compute model value
    m = (
        f
        + g @ s
        + 0.5 * s @ H @ s
        + third_order_term
        + 0.25 * sigma * (LA.norm(s) ** 4)
    )

    return m


def ar3_model_gradient(s, g, H, T, sigma):
    """
    Compute the gradient of the AR3 model.

    ∇m(s) = g + H*s + (1/2)*T*s + sigma*||s||^2*s

    Parameters:
    - s: Step vector (n, 1) or (n,)
    - g: Gradient at current point (n, 1) or (n,)
    - H: Hessian at current point (n, n)
    - T: Third-order derivative tensor (n, n, n)
    - sigma: Regularization parameter (scalar)

    Returns:
    - grad_m: Gradient of model (n,)
    """
    s = s.flatten()
    g = g.flatten()

    # Compute T*s (results in n×n matrix)
    T_s = np.tensordot(T, s, axes=([0], [0]))

    # Model gradient
    grad_m = g + H @ s + 0.5 * T_s @ s + sigma * (LA.norm(s) ** 2) * s

    return grad_m


def ar3_model_hessian(s, H, T, sigma):
    """
    Compute the Hessian of the AR3 model.

    ∇²m(s) = H + T*s + sigma*(||s||^2*I + 2*s*s')

    Parameters:
    - s: Step vector (n, 1) or (n,)
    - H: Hessian at current point (n, n)
    - T: Third-order derivative tensor (n, n, n)
    - sigma: Regularization parameter (scalar)

    Returns:
    - hess_m: Hessian of model (n, n)
    """
    s = s.flatten()
    n = len(s)

    # Compute T*s
    T_s = np.tensordot(T, s, axes=([0], [0]))

    # Model Hessian
    norm_s_sq = LA.norm(s) ** 2
    hess_m = H + T_s + sigma * (norm_s_sq * np.eye(n) + 2 * np.outer(s, s))

    return hess_m


def solve_ar3_subproblem(g, H, T, sigma, f=0, method="trust-constr", max_iter=1000):
    """
    Solve the AR3 subproblem using scipy.

    minimize m(s) = f + g'*s + (1/2)*s'*H*s + (1/6)*s'*T*s + (1/4)*sigma*||s||^4

    Parameters:
    - g: Gradient (n, 1) or (n,)
    - H: Hessian (n, n)
    - T: Third-order tensor (n, n, n)
    - sigma: Regularization parameter
    - f: Function value (for model evaluation, default 0)
    - method: Optimization method for scipy.minimize
    - max_iter: Maximum iterations for subproblem solver

    Returns:
    - s: Step vector (n, 1)
    - success: Whether subproblem was solved successfully
    - n_iter: Number of iterations used
    """
    n = len(g) if len(g.shape) == 1 else g.shape[0]
    g = g.flatten()

    # Define objective and its derivatives for scipy
    def objective(s):
        return ar3_model(s, f, g, H, T, sigma)

    def gradient(s):
        return ar3_model_gradient(s, g, H, T, sigma)

    def hessian(s):
        return ar3_model_hessian(s, H, T, sigma)

    # Initial guess
    s0 = np.zeros(n)

    # Solve using scipy.optimize.minimize
    try:
        if method == "trust-constr":
            result = minimize(
                objective,
                s0,
                method=method,
                jac=gradient,
                hess=hessian,
                options={"maxiter": max_iter, "verbose": 0},
            )
        else:
            result = minimize(
                objective,
                s0,
                method=method,
                jac=gradient,
                options={"maxiter": max_iter},
            )

        s = result.x.reshape(-1, 1)
        success = result.success
        n_iter = result.nit if hasattr(result, "nit") else max_iter

    except Exception as e:
        # If solver fails, return zero step
        s = np.zeros((n, 1))
        success = False
        n_iter = 0

    return s, success, n_iter


def solve_ar3_subproblem_ar2(
    g,
    H,
    T,
    sigma,
    f=0,
    max_iter_outer=100,
    eta1=0.01,
    eta2=0.95,
    gamma1=0.5,
    gamma2=3.0,
    sigma_min_inner=1e-8,
    verbose_inner=False,
):
    """
    Solve the AR3 subproblem using AR2-Simple as the inner solver.

    This implements the approach described in the paper: using AR2 with Simple Update
    to minimize the AR3 cubic model.

    minimize m(s) = f + g'*s + (1/2)*s'*H*s + (1/6)*s'*T*s + (1/4)*sigma*||s||^4

    Parameters:
    - g: Gradient (n, 1) or (n,)
    - H: Hessian (n, n)
    - T: Third-order tensor (n, n, n)
    - sigma: AR3 regularization parameter (for the ||s||^4 term)
    - f: Function value (for model evaluation, default 0)
    - max_iter_outer: Maximum iterations for AR2
    - eta1, eta2, gamma1, gamma2: AR2 Simple Update parameters
    - sigma_min_inner: Minimum sigma for inner AR2 solver
    - verbose_inner: Print AR2 solver details

    Returns:
    - s: Step vector (n, 1)
    - success: Whether subproblem was solved successfully
    - n_iter: Number of iterations used
    """
    if not AR2_AVAILABLE:
        raise ImportError(
            "AR2 module not available. Cannot use AR2 as subproblem solver."
        )

    n = len(g) if len(g.shape) == 1 else g.shape[0]
    g = g.flatten()

    # Define AR3 model as objective for AR2
    def ar3_objective(s):
        return ar3_model(s, f, g, H, T, sigma)

    def ar3_gradient(s):
        return ar3_model_gradient(s, g, H, T, sigma)

    def ar3_hessian(s):
        return ar3_model_hessian(s, H, T, sigma)

    # Initial guess
    s0 = np.zeros(n)

    # Estimate initial inner sigma based on problem scale
    # Use Frobenius norm of third derivative as a heuristic
    # Note: T is a 3D tensor, so we flatten it to compute the norm
    sigma0_inner = max(0.1 * np.linalg.norm(T.flatten()), sigma_min_inner)

    # Solve using AR2-Simple
    try:
        result = ar2_simple(
            model_func=ar3_objective,
            model_grad=ar3_gradient,
            model_hess=ar3_hessian,
            s0=s0,
            sigma0=sigma0_inner,
            eta1=eta1,
            eta2=eta2,
            gamma1=gamma1,
            gamma2=gamma2,
            sigma_min=sigma_min_inner,
            max_iterations=max_iter_outer,
            tol=1e-6,
            verbose=verbose_inner,
        )

        s = result["s_final"]
        success = result["converged"]
        n_iter = result["iterations"]

    except Exception as e:
        # If solver fails, return zero step
        s = np.zeros((n, 1))
        success = False
        n_iter = 0

    return s, success, n_iter


def analyze_persistent_minimizer(taylor_coeffs, step_norm, tolerance=0):
    """
    Check if step_norm is a persistent minimizer of the Taylor polynomial.

    This implements the "pre-rejection" mechanism of AR3.

    Parameters:
    - taylor_coeffs: Coefficients of Taylor polynomial [c3, c2, c1, c0]
                     representing c3*α³ + c2*α² + c1*α + c0
    - step_norm: The value to check (α = ||s||)
    - tolerance: Tolerance for derivative check

    Returns:
    - persistent_alpha: The persistent minimizer location (or 0 if transient)
    """
    if len(taylor_coeffs) < 2:
        return step_norm

    # Compute derivative of Taylor polynomial
    derivative_coeffs = np.polyder(taylor_coeffs)

    # Find roots of derivative in [0, step_norm]
    try:
        roots = np.roots(derivative_coeffs)
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        valid_roots = real_roots[(real_roots >= 0) & (real_roots <= step_norm + 1e-10)]

        if len(valid_roots) == 0:
            return 0  # Transient

        # Check second derivative at valid roots to find minima
        second_derivative_coeffs = np.polyder(derivative_coeffs)

        for root in valid_roots:
            second_deriv = np.polyval(second_derivative_coeffs, root)
            if second_deriv > tolerance:  # Local minimum
                return float(root)

        return 0  # No valid persistent minimizer
    except:
        # If root finding fails, assume persistent
        return step_norm


def ar3(
    fx,
    dx,
    d2x,
    d3x,
    x0,
    max_iterations,
    tol,
    param_list,
    verbose=True,
    subproblem_solver="scipy",
):
    """
    AR3 (Adaptive Regularization using Third-order derivatives) Algorithm.

    This implementation uses the Simple Update strategy for sigma.

    Parameters:
    - fx: Objective function f(x): R^n -> R
    - dx: Gradient function ∇f(x): R^n -> R^n
    - d2x: Hessian function ∇²f(x): R^n -> R^(n×n)
    - d3x: Third-order derivative function ∇³f(x): R^n -> R^(n×n×n)
    - x0: Initial point (numpy array of shape (n, 1))
    - max_iterations: Maximum number of iterations (int)
    - tol: Convergence tolerance for ||∇f(x)|| (float)
    - param_list: List of parameters [eta1, eta2, gamma1, gamma2, sigma_min] where:
        - eta1: Successful cutoff (accept step if ρ >= eta1), default 0.01
        - eta2: Very successful cutoff (reduce sigma if ρ >= eta2), default 0.95
        - gamma1: Sigma decrease factor, default 0.5
        - gamma2: Sigma increase factor, default 3.0
        - sigma_min: Minimum sigma value, default 1e-8
    - verbose: Print iteration details (default True)
    - subproblem_solver: Subproblem solver to use, options:
        - 'scipy': Use scipy.optimize.minimize (default)
        - 'ar2': Use AR2-Simple as inner solver (as described in paper)

    Returns:
    - dict: Dictionary containing:
        - 'x_final': Final point (numpy array of shape (n, 1))
        - 'iterations': Number of iterations performed (int)
        - 'x_history': List of x_k at each iteration
        - 's_history': List of steps s_k at each iteration
        - 'sigma_history': List of sigma_k at each iteration
        - 'f_history': List of f(x_k) at each iteration
        - 'success_history': List of success flags at each iteration
        - 'grad_norm_history': List of ||∇f(x_k)|| at each iteration
        - 'rho_history': List of ρ_k values
        - 'subproblem_solves': Total number of subproblem solves
        - 'n_successful': Number of successful iterations
        - 'n_unsuccessful': Number of unsuccessful iterations
        - 'converged': Boolean indicating if convergence was achieved
    """
    # Unpack parameters
    if len(param_list) >= 5:
        eta1, eta2, gamma1, gamma2, sigma_min = param_list[:5]
    else:
        # Default parameters
        eta1 = 0.01
        eta2 = 0.95
        gamma1 = 0.5
        gamma2 = 3.0
        sigma_min = 1e-8

    # Validate parameters
    if not (0 < eta1 < eta2 < 1 and 0 < gamma1 < 1 and gamma2 > 1 and sigma_min > 0):
        raise ValueError(
            "Parameters must satisfy: 0 < eta1 < eta2 < 1, 0 < gamma1 < 1, gamma2 > 1, sigma_min > 0"
        )

    # Initialize
    x_k = x0.copy()
    k = 0
    n = x0.shape[0]

    # Estimate initial sigma (simple heuristic)
    g0 = dx(x0)
    H0 = d2x(x0)
    sigma = max(1.0, sigma_min)  # Simple initial guess

    # Counters
    subproblem_counter = 0

    # History tracking
    x_history = [x0.copy()]
    f_history = [fx(x0)]
    s_history = []
    sigma_history = [sigma]
    success_history = []
    grad_norm_history = [LA.norm(g0)]
    rho_history = []

    if verbose:
        print("=" * 100)
        print("AR3 Algorithm - Simple Update")
        print("=" * 100)
        print(f"Parameters: eta1={eta1}, eta2={eta2}, gamma1={gamma1}, gamma2={gamma2}")
        print(f"Initial sigma: {sigma}")
        print(f"Subproblem solver: {subproblem_solver}")
        if subproblem_solver == "ar2" and not AR2_AVAILABLE:
            print(
                "WARNING: AR2 solver requested but not available, falling back to scipy"
            )
        print()

    while k < max_iterations:
        # Step 1: Compute derivatives
        f_k = fx(x_k)
        g_k = dx(x_k)
        H_k = d2x(x_k)
        T_k = d3x(x_k)
        grad_norm = LA.norm(g_k)

        # Step 2: Check termination
        if grad_norm <= tol:
            if verbose:
                print(f"Converged at iteration {k}, ||∇f(x)|| = {grad_norm:.6e}")
            break

        if verbose:
            print(f"Iteration {k}:")
            print(f"  x_k: {x_k.flatten()}")
            print(f"  f(x_k): {f_k:.6e}")
            print(f"  ||∇f(x_k)||: {grad_norm:.6e}")
            print(f"  sigma: {sigma:.6e}")

        # Step 3: Solve subproblem
        if subproblem_solver == "ar2":
            s_k, sub_success, sub_iter = solve_ar3_subproblem_ar2(
                g_k,
                H_k,
                T_k,
                sigma,
                f_k,
                max_iter_outer=100,
                eta1=eta1,
                eta2=eta2,
                gamma1=gamma1,
                gamma2=gamma2,
                sigma_min_inner=sigma_min,
                verbose_inner=False,
            )
        else:  # default to 'scipy'
            s_k, sub_success, sub_iter = solve_ar3_subproblem(g_k, H_k, T_k, sigma, f_k)

        subproblem_counter += 1
        norm_s = LA.norm(s_k)

        if verbose:
            print(f"  ||s_k||: {norm_s:.6e}")
            print(f"  Subproblem success: {sub_success}")

        # Step 4: Compute predicted decrease (using Taylor expansion)
        m_0 = f_k
        m_s = ar3_model(s_k, f_k, g_k, H_k, T_k, 0)  # Taylor part only
        predicted_decrease = m_0 - m_s

        # Ensure predicted decrease is non-negative
        if predicted_decrease < 0:
            predicted_decrease = abs(predicted_decrease)

        # Step 5: Compute actual decrease
        f_trial = fx(x_k + s_k)
        actual_decrease = f_k - f_trial

        # Step 6: Compute acceptance ratio
        eps = np.finfo(float).eps * max(1, abs(f_k))
        if predicted_decrease > eps:
            rho = actual_decrease / predicted_decrease
        else:
            # Predicted decrease too small - numerical issues
            rho = -np.inf

        rho_history.append(rho)

        if verbose:
            print(f"  Predicted decrease: {predicted_decrease:.6e}")
            print(f"  Actual decrease: {actual_decrease:.6e}")
            print(f"  rho: {rho:.6e}")

        # Step 7: Update iterate and sigma
        if rho >= eta2:
            # Very successful iteration
            x_k = x_k + s_k
            sigma = max(gamma1 * sigma, sigma_min)
            success = True
            if verbose:
                print(
                    f"  Very successful! Accepting step, decreasing sigma to {sigma:.6e}"
                )
        elif rho >= eta1:
            # Successful iteration
            x_k = x_k + s_k
            sigma = max(sigma, sigma_min)
            success = True
            if verbose:
                print(f"  Successful! Accepting step, keeping sigma = {sigma:.6e}")
        else:
            # Unsuccessful iteration
            # x_k remains unchanged
            sigma = gamma2 * sigma
            success = False
            if verbose:
                print(
                    f"  Unsuccessful. Rejecting step, increasing sigma to {sigma:.6e}"
                )

        # Record history
        s_history.append(s_k.copy())
        success_history.append(success)
        x_history.append(x_k.copy())
        f_history.append(fx(x_k))
        grad_norm_history.append(LA.norm(dx(x_k)))
        sigma_history.append(sigma)

        if verbose:
            print()

        k += 1

    if k >= max_iterations and verbose:
        print(f"Reached maximum iterations {max_iterations}, did not converge")

    # Compute final statistics
    n_successful = sum(1 for s in success_history if s == True)
    n_unsuccessful = len(success_history) - n_successful

    if verbose:
        print("=" * 100)
        print(f"Final results:")
        print(f"  Iterations: {k}")
        print(f"  Successful: {n_successful}, Unsuccessful: {n_unsuccessful}")
        print(f"  Subproblem solves: {subproblem_counter}")
        print(f"  Final ||∇f(x)||: {grad_norm_history[-1]:.6e}")
        print(f"  Final f(x): {f_history[-1]:.6e}")
        print("=" * 100)

    return {
        "x_final": x_k,
        "iterations": k,
        "x_history": x_history,
        "s_history": s_history,
        "sigma_history": sigma_history,
        "f_history": f_history,
        "success_history": success_history,
        "grad_norm_history": grad_norm_history,
        "rho_history": rho_history,
        "subproblem_solves": subproblem_counter,
        "n_successful": n_successful,
        "n_unsuccessful": n_unsuccessful,
        "converged": k < max_iterations,
    }


# ============================================================================
# AR3-Interp+: AR3 with Interpolation Update and Pre-rejection
# ============================================================================


def padded_polyder(poly):
    """
    Compute polynomial derivative with zero-padding to maintain length.

    Parameters:
    - poly: Polynomial coefficients (highest degree first)

    Returns:
    - der: Derivative coefficients (padded to length len(poly)-1)
    """
    der = np.polyder(poly)
    if len(der) < len(poly) - 1:
        der = np.concatenate([np.zeros(len(poly) - 1 - len(der)), der])
    return der


def real_roots(poly):
    """
    Find real roots of a polynomial.

    Parameters:
    - poly: Polynomial coefficients

    Returns:
    - roots: Array of real roots
    """
    roots_all = np.roots(poly)
    # Filter real roots (imaginary part is negligible)
    real_roots_arr = roots_all[np.abs(roots_all.imag) < 1e-10].real
    return real_roots_arr


def analyze_persistent_min_ar3(taylor_poly, tolerance=0):
    """
    Analyze persistent minimizers of a Taylor polynomial (for pre-rejection).

    This implements the pre-rejection mechanism described in the paper.

    Parameters:
    - taylor_poly: Coefficients of Taylor polynomial (highest degree first)
    - tolerance: Tolerance for persistence check

    Returns:
    - persistent_alpha: Boundary of persistent region (or -1 if transient)
    """
    p = len(taylor_poly) - 1

    # Check if it's a descent direction
    if taylor_poly[-2] >= 0:  # Linear term coefficient
        return -1

    # Compute derivative polynomials
    taylor_der_poly = padded_polyder(taylor_poly)

    # pos_sigma_poly: tolerance - taylor_der_poly
    pos_sigma_poly = np.concatenate([np.zeros(p - 1), [tolerance]]) - taylor_der_poly

    # local_min_poly: [polyder(taylor_der_poly), 0] + p * pos_sigma_poly
    # Note: pad with 0 BEFORE adding
    local_min_poly = (
        np.concatenate([padded_polyder(taylor_der_poly), [0]]) + p * pos_sigma_poly
    )

    # Find candidate alphas from pos_sigma_poly roots
    alpha_options1 = real_roots(pos_sigma_poly)
    if len(alpha_options1) > 0:
        # Filter by local_min_poly constraint
        local_min_vals = np.polyval(local_min_poly, alpha_options1)
        alpha_options1 = alpha_options1[local_min_vals >= -10 * np.finfo(float).eps]

    # Find candidate alphas from local_min_poly roots
    alpha_options2 = real_roots(local_min_poly)
    if len(alpha_options2) > 0:
        # Filter by pos_sigma_poly constraint
        pos_sigma_vals = np.polyval(pos_sigma_poly, alpha_options2)
        alpha_options2 = alpha_options2[pos_sigma_vals >= -10 * np.finfo(float).eps]

    # Combine and filter positive values
    alpha_options = (
        np.concatenate([alpha_options1, alpha_options2])
        if len(alpha_options1) > 0 or len(alpha_options2) > 0
        else np.array([])
    )
    alpha_options = alpha_options[alpha_options > 0]

    # Return minimum positive alpha (or inf if none)
    persistent_alpha = np.min(alpha_options) if len(alpha_options) > 0 else np.inf

    return persistent_alpha


def pos_boundary_points(poly_constraints):
    """
    Compute positive boundary points of feasible set defined by polynomial constraints.

    All constraints are of the form: polyval(poly, x) <= 0

    Parameters:
    - poly_constraints: List of polynomial constraint coefficients

    Returns:
    - boundary: Array of boundary points
    """
    num_constraints = len(poly_constraints)
    boundary = []

    for i in range(num_constraints):
        # Find roots of constraint i
        points = real_roots(poly_constraints[i])
        points = points[points > 0]

        # Check if points satisfy all other constraints
        for j in list(range(i)) + list(range(i + 1, num_constraints)):
            if len(points) > 0:
                vals = np.polyval(poly_constraints[j], points)
                points = points[vals <= 10 * np.finfo(float).eps]

        if len(points) > 0:
            boundary.extend(points.tolist())

    return np.array(boundary)


def optimize_sigma_ar3(
    taylor_poly,
    prev_sigma,
    norm_step,
    constraint_poly,
    decrease_sigma,
    use_prerejection=True,
):
    """
    Find optimal sigma satisfying interpolation constraints.

    Parameters:
    - taylor_poly: Taylor polynomial coefficients
    - prev_sigma: Previous sigma value
    - norm_step: Norm of the step
    - constraint_poly: Constraint polynomial
    - decrease_sigma: If True, find largest sigma <= prev_sigma; else smallest >= prev_sigma
    - use_prerejection: Whether to use pre-rejection mechanism

    Returns:
    - sigma: Optimal sigma value (or nan if not found)
    - alpha: Corresponding alpha value (or nan)
    """
    p = len(taylor_poly) - 1
    taylor_der_poly = padded_polyder(taylor_poly)

    # Compute sigma as a function of alpha
    def compute_sigma(alpha):
        return -np.polyval(taylor_der_poly, alpha) / (norm_step ** (p + 1) * alpha**p)

    # Build constraint polynomials
    # prev_sigma_poly: constraint based on prev_sigma
    sign = -1 if decrease_sigma else 1
    prev_sigma_poly = sign * np.concatenate(
        [[prev_sigma * norm_step ** (p + 1)], taylor_der_poly]
    )

    # Collect all constraints
    constraints = [constraint_poly, prev_sigma_poly]

    if use_prerejection:
        alpha_persistent = analyze_persistent_min_ar3(taylor_poly)
        if alpha_persistent < np.inf:
            # Add constraint: alpha <= alpha_persistent, i.e., alpha - alpha_persistent <= 0
            constraints.append(np.array([1, -alpha_persistent]))
    else:
        # Without pre-rejection, add standard constraints
        local_min_poly_temp = -padded_polyder(taylor_der_poly) + p * taylor_der_poly
        local_min_poly = np.concatenate([local_min_poly_temp, [0]])
        constraints.extend([taylor_der_poly, local_min_poly])

    # Find boundary points
    alpha_options = pos_boundary_points(constraints)

    if len(alpha_options) == 0:
        return np.nan, np.nan

    # Select best alpha and sigma
    sigma_options = compute_sigma(alpha_options)
    sigma_options = np.maximum(sigma_options, 0)

    if decrease_sigma:
        best_idx = np.argmax(sigma_options)
    else:
        best_idx = np.argmin(sigma_options)

    sigma = sigma_options[best_idx]
    alpha = alpha_options[best_idx]

    return sigma, alpha


def construct_taylor_poly_ar3(s, f, g, H, T):
    """
    Construct 1D Taylor polynomial along step direction s.

    Returns polynomial t(alpha) = f(x + alpha*s) (Taylor approximation)
    Coefficients are [c3, c2, c1, c0] for c3*alpha^3 + c2*alpha^2 + c1*alpha + c0

    Parameters:
    - s: Step direction (n,) or (n, 1)
    - f: Function value
    - g: Gradient (n,) or (n, 1)
    - H: Hessian (n, n)
    - T: Third-order tensor (n, n, n)

    Returns:
    - taylor_poly: Polynomial coefficients [c3, c2, c1, c0]
    """
    s = s.flatten()
    g = g.flatten()

    # Compute third-order term coefficient
    T_s = np.tensordot(T, s, axes=([0], [0]))
    c3 = (1 / 6) * s @ T_s @ s

    # Second-order term
    c2 = 0.5 * s @ H @ s

    # Linear term
    c1 = g @ s

    # Constant term
    c0 = f

    return np.array([c3, c2, c1, c0])


def ar3_interp(
    fx,
    dx,
    d2x,
    d3x,
    x0,
    max_iterations,
    tol,
    param_list=None,
    verbose=True,
    subproblem_solver="scipy",
):
    """
    AR3-Interp+: AR3 with Interpolation Update and Pre-rejection.

    This is the best-performing AR3 variant from the paper, combining:
    1. Interpolation update for sigma (more adaptive than Simple Update)
    2. Pre-rejection mechanism (avoids transient minimizers)

    Parameters:
    - fx: Objective function f(x): R^n -> R
    - dx: Gradient function ∇f(x): R^n -> R^n
    - d2x: Hessian function ∇²f(x): R^n -> R^(n×n)
    - d3x: Third derivative tensor function: R^n -> R^(n×n×n)
    - x0: Initial point (n, 1) or (n,)
    - max_iterations: Maximum number of iterations
    - tol: Gradient norm tolerance for convergence
    - param_list: [eta1, eta2, gamma1, gamma3, beta, alpha_max, sigma_min]
                  eta1: successful cutoff (default: 0.01)
                  eta2: very successful cutoff (default: 0.95)
                  gamma1: sigma decrease factor (default: 0.5)
                  gamma3: large decrease factor (default: 0.1)
                  beta: interpolation reduction factor (default: 0.01)
                  alpha_max: maximum alpha for interpolation (default: 2.0)
                  sigma_min: minimum sigma (default: 1e-8)
    - verbose: Print iteration information
    - subproblem_solver: 'scipy' or 'ar2'

    Returns:
    - dict: Results dictionary with convergence history
    """
    # Parse parameters
    if param_list is None:
        eta1 = 0.01
        eta2 = 0.95
        gamma1 = 0.5
        gamma3 = 0.1
        beta = 0.01
        alpha_max = 2.0
        sigma_min = 1e-8
    else:
        if len(param_list) >= 7:
            eta1, eta2, gamma1, gamma3, beta, alpha_max, sigma_min = param_list[:7]
        else:
            raise ValueError("param_list should have at least 7 elements")

    # Validate parameters
    if not (0 < eta1 < eta2 < 1 and 0 < gamma1 < 1 and 0 < gamma3 < 1):
        raise ValueError(
            "Parameters must satisfy: 0 < eta1 < eta2 < 1, 0 < gamma1 < 1, 0 < gamma3 < 1"
        )

    # Initialize
    x_k = x0.copy().reshape(-1, 1)
    k = 0
    n = x_k.shape[0]

    # Estimate initial sigma
    g0 = dx(x0)
    sigma = max(1.0, sigma_min)

    # Counters
    subproblem_counter = 0
    pre_rejections = 0
    interp_updates = 0

    # History tracking
    x_history = [x0.copy()]
    s_history = []
    f_history = [fx(x0)]
    grad_norm_history = [LA.norm(g0)]
    sigma_history = [sigma]
    rho_history = []
    success_history = []
    persistent_alpha_history = []

    if verbose:
        print("=" * 100)
        print("AR3-Interp+ (AR3 with Interpolation Update and Pre-rejection)")
        print("=" * 100)
        print(f"Parameters: eta1={eta1}, eta2={eta2}, gamma1={gamma1}, gamma3={gamma3}")
        print(f"            beta={beta}, alpha_max={alpha_max}, sigma_min={sigma_min}")
        print(f"Initial sigma: {sigma}")
        print(f"Subproblem solver: {subproblem_solver}")
        if subproblem_solver == "ar2" and not AR2_AVAILABLE:
            print(
                "WARNING: AR2 solver requested but not available, falling back to scipy"
            )
        print()

    while k < max_iterations:
        # Step 1: Compute derivatives
        f_k = fx(x_k)
        g_k = dx(x_k)
        H_k = d2x(x_k)
        T_k = d3x(x_k)
        grad_norm = LA.norm(g_k)

        # Step 2: Check termination
        if grad_norm <= tol:
            if verbose:
                print(f"Converged at iteration {k}, ||∇f(x)|| = {grad_norm:.6e}")
            break

        if verbose:
            print(f"Iteration {k}:")
            print(f"  f(x_k): {f_k:.6e}")
            print(f"  ||∇f(x_k)||: {grad_norm:.6e}")
            print(f"  sigma: {sigma:.6e}")

        # Step 3: Solve subproblem
        if subproblem_solver == "ar2":
            s_k, sub_success, sub_iter = solve_ar3_subproblem_ar2(
                g_k,
                H_k,
                T_k,
                sigma,
                f_k,
                max_iter_outer=100,
                eta1=eta1,
                eta2=eta2,
                gamma1=gamma1,
                gamma2=3.0,
                sigma_min_inner=sigma_min,
                verbose_inner=False,
            )
        else:
            s_k, sub_success, sub_iter = solve_ar3_subproblem(g_k, H_k, T_k, sigma, f_k)

        subproblem_counter += 1
        norm_s = LA.norm(s_k)

        if verbose:
            print(f"  ||s_k||: {norm_s:.6e}")

        # Step 4: Construct Taylor polynomial along step direction
        taylor_poly = construct_taylor_poly_ar3(s_k, f_k, g_k, H_k, T_k)

        # Step 5: Compute model value with regularization
        model_poly_0 = np.polyval(taylor_poly, 0)
        model_poly_1 = np.polyval(taylor_poly, 1)

        # Add regularization term to model
        model_1_with_reg = model_poly_1 + 0.25 * sigma * (norm_s**4)

        # Predicted decrease (using Taylor approximation)
        predicted_decrease = f_k - model_poly_1

        if predicted_decrease < 0:
            predicted_decrease = abs(predicted_decrease)

        # Step 6: Pre-rejection check
        transient = False
        persistent_alpha = np.inf

        if norm_s > 1e-15:  # Only check if step is non-trivial
            persistent_alpha = analyze_persistent_min_ar3(taylor_poly, tolerance=0)
            transient = persistent_alpha < 1 - 10 * np.finfo(float).eps

            if transient:
                pre_rejections += 1
                if verbose:
                    print(
                        f"  Pre-rejection: transient minimizer (persistent_alpha={persistent_alpha:.6e})"
                    )

        persistent_alpha_history.append(persistent_alpha)

        # Step 7: Compute actual decrease (only if not pre-rejected)
        if not transient:
            f_trial = fx(x_k + s_k)
            actual_decrease = f_k - f_trial

            # Compute acceptance ratio
            eps_val = np.finfo(float).eps * max(1, abs(f_k))
            if predicted_decrease > eps_val:
                rho = actual_decrease / predicted_decrease
            else:
                rho = -np.inf
        else:
            # Pre-rejected: set rho to nan
            rho = np.nan
            actual_decrease = np.nan

        rho_history.append(rho)

        if verbose and not transient:
            print(f"  Predicted decrease: {predicted_decrease:.6e}")
            print(f"  Actual decrease: {actual_decrease:.6e}")
            print(f"  rho: {rho:.6e}")

        # Step 8: Update iterate and sigma using Interpolation Update
        success = False

        if not transient and rho >= 1 + np.finfo(float).eps:
            # Extremely successful: use interpolation to decrease sigma
            # Construct interpolation polynomial
            f_plus = f_k - actual_decrease
            interp_poly = np.concatenate([[f_plus - model_poly_1], taylor_poly])

            # Compute constraint for successful_sigma
            taylor_der_poly = padded_polyder(taylor_poly)
            p = len(taylor_poly) - 1
            model_poly = np.concatenate([[0], taylor_poly]) - np.concatenate(
                [[0], (1 / (p + 1)) * taylor_der_poly, [0]]
            )

            # Determine which polynomial to use for difference
            if np.polyval(interp_poly, 1) >= np.polyval(taylor_poly, 1):
                diff_poly = model_poly - interp_poly
            else:
                diff_poly = model_poly - np.concatenate([[0], taylor_poly])

            current_diff = np.polyval(diff_poly, 1)
            min_success_diff = 1e-8

            if current_diff >= min_success_diff:
                # Use interpolation
                constraint_poly = diff_poly - np.concatenate(
                    [np.zeros(p + 1), [beta * current_diff]]
                )
                new_sigma, alpha_opt = optimize_sigma_ar3(
                    taylor_poly,
                    sigma,
                    norm_s,
                    constraint_poly,
                    decrease_sigma=True,
                    use_prerejection=True,
                )

                if not np.isnan(alpha_opt) and alpha_opt <= alpha_max:
                    sigma = new_sigma
                    interp_updates += 1
                else:
                    sigma = gamma3 * sigma
            else:
                sigma = gamma1 * sigma

            sigma = max(sigma, sigma_min)
            x_k = x_k + s_k
            success = True

            if verbose:
                print(f"  Extremely successful! Accepting step, sigma={sigma:.6e}")

        elif not transient and rho >= eta2:
            # Very successful
            sigma = gamma1 * sigma
            sigma = max(sigma, sigma_min)
            x_k = x_k + s_k
            success = True

            if verbose:
                print(f"  Very successful! Accepting step, sigma={sigma:.6e}")

        elif not transient and rho >= eta1:
            # Successful
            sigma = max(sigma, sigma_min)
            x_k = x_k + s_k
            success = True

            if verbose:
                print(f"  Successful! Accepting step, sigma={sigma:.6e}")

        elif transient or rho >= -np.finfo(float).eps:
            # Unsuccessful or pre-rejected
            sigma = 3.0 * sigma  # Simple increase (could use interpolation here too)

            if verbose:
                if transient:
                    print(
                        f"  Pre-rejected (transient). Increasing sigma to {sigma:.6e}"
                    )
                else:
                    print(
                        f"  Unsuccessful. Rejecting step, increasing sigma to {sigma:.6e}"
                    )

        else:
            # Extremely unsuccessful: use interpolation to increase sigma
            if not np.isnan(actual_decrease):
                f_plus = f_k - actual_decrease
                interp_poly = np.concatenate([[f_plus - model_poly_1], taylor_poly])

                # Build constraint for unsuccessful_sigma
                # actual_decrease_poly: [-interp_poly[:-1], 0]
                actual_decrease_poly = np.concatenate([-interp_poly[:-1], [0]])
                predicted_decrease_poly = np.concatenate([[0], -taylor_poly[:-1], [0]])
                constraint_poly = actual_decrease_poly - eta1 * predicted_decrease_poly
                constraint_poly = constraint_poly[:-1]  # Remove constant term

                new_sigma, _ = optimize_sigma_ar3(
                    taylor_poly,
                    sigma,
                    norm_s,
                    -constraint_poly,
                    decrease_sigma=False,
                    use_prerejection=True,
                )

                if not np.isnan(new_sigma) and new_sigma >= 3.0 * sigma:
                    sigma = min(new_sigma, 100 * sigma)  # Cap at max_increase
                    interp_updates += 1
                else:
                    sigma = 3.0 * sigma
            else:
                sigma = 3.0 * sigma

            if verbose:
                print(f"  Extremely unsuccessful. Rejecting step, sigma={sigma:.6e}")

        # Record history
        s_history.append(s_k.copy())
        x_history.append(x_k.copy())
        f_history.append(fx(x_k))
        grad_norm_history.append(LA.norm(dx(x_k)))
        sigma_history.append(sigma)
        success_history.append(success)

        if verbose:
            print()

        k += 1

    if k >= max_iterations and verbose:
        print(f"Reached maximum iterations {max_iterations}, did not converge")

    # Compute final statistics
    n_successful = sum(success_history)
    n_unsuccessful = len(success_history) - n_successful

    if verbose:
        print("=" * 100)
        print("AR3-Interp+ Summary:")
        print(f"  Converged: {k < max_iterations}")
        print(f"  Iterations: {k}")
        print(f"  Successful iterations: {n_successful}")
        print(f"  Unsuccessful iterations: {n_unsuccessful}")
        print(f"  Pre-rejections: {pre_rejections}")
        print(f"  Interpolation updates: {interp_updates}")
        print(f"  Subproblem solves: {subproblem_counter}")
        print(f"  Final ||∇f(x)||: {grad_norm_history[-1]:.6e}")
        print(f"  Final f(x): {f_history[-1]:.6e}")
        print("=" * 100)

    return {
        "x_final": x_k,
        "iterations": k,
        "x_history": x_history,
        "s_history": s_history,
        "sigma_history": sigma_history,
        "f_history": f_history,
        "success_history": success_history,
        "grad_norm_history": grad_norm_history,
        "rho_history": rho_history,
        "persistent_alpha_history": persistent_alpha_history,
        "subproblem_solves": subproblem_counter,
        "pre_rejections": pre_rejections,
        "interp_updates": interp_updates,
        "n_successful": n_successful,
        "n_unsuccessful": n_unsuccessful,
        "converged": k < max_iterations,
    }
