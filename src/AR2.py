"""
AR2 (Adaptive Regularization using Second-order derivatives) Algorithm

Python implementation of the AR2 algorithm, primarily used as a subproblem solver for AR3.

Includes:
- ar2_simple: Basic AR2 with Simple Update (for use as inner solver)
- ar2_interp: AR2 with Interpolation Update (AR2-Interp, best-performing AR2 variant)
"""

import numpy as np
import numpy.linalg as LA
from scipy.optimize import minimize


def ar2_model(s, f, g, H, sigma):
    """
    Compute the AR2 regularized quadratic model value.

    Model: m(s) = f + g'*s + (1/2)*s'*H*s + (1/3)*sigma*||s||^3

    Parameters:
    - s: Step vector (n, 1) or (n,)
    - f: Function value at current point (scalar)
    - g: Gradient at current point (n, 1) or (n,)
    - H: Hessian at current point (n, n)
    - sigma: Regularization parameter (scalar)

    Returns:
    - m: Model value (scalar)
    """
    s = s.flatten()
    g = g.flatten()

    m = f + g @ s + 0.5 * s @ H @ s + (1 / 3) * sigma * (LA.norm(s) ** 3)

    return m


def ar2_model_gradient(s, g, H, sigma):
    """
    Compute the gradient of the AR2 model.

    ∇m(s) = g + H*s + sigma*||s||*s

    Parameters:
    - s: Step vector (n, 1) or (n,)
    - g: Gradient at current point (n, 1) or (n,)
    - H: Hessian at current point (n, n)
    - sigma: Regularization parameter (scalar)

    Returns:
    - grad_m: Gradient of model (n,)
    """
    s = s.flatten()
    g = g.flatten()

    norm_s = LA.norm(s)

    if norm_s > 0:
        grad_m = g + H @ s + sigma * norm_s * s
    else:
        grad_m = g

    return grad_m


def ar2_model_hessian(s, H, sigma):
    """
    Compute the Hessian of the AR2 model.

    ∇²m(s) = H + sigma*(||s||*I + s*s'/||s||)

    Parameters:
    - s: Step vector (n, 1) or (n,)
    - H: Hessian at current point (n, n)
    - sigma: Regularization parameter (scalar)

    Returns:
    - hess_m: Hessian of model (n, n)
    """
    s = s.flatten()
    n = len(s)
    norm_s = LA.norm(s)

    if norm_s > 1e-10:
        hess_m = H + sigma * (norm_s * np.eye(n) + np.outer(s, s) / norm_s)
    else:
        hess_m = H

    return hess_m


def solve_ar2_newton_step(g, H, sigma):
    """
    Solve one Newton step for the AR2 model.

    Solve: ∇²m(s_k) * delta_s = -∇m(s_k)

    Parameters:
    - g: Current gradient of AR2 model (n,)
    - H: Current Hessian of AR2 model (n, n)
    - sigma: Regularization parameter

    Returns:
    - delta_s: Newton step (n,)
    - success: Whether the step was computed successfully
    """
    try:
        # Try Cholesky decomposition (for positive definite Hessian)
        L = LA.cholesky(H)
        delta_s = -LA.solve(L @ L.T, g)
        success = True
    except LA.LinAlgError:
        # If not positive definite, use pseudo-inverse
        try:
            delta_s = -LA.lstsq(H, g, rcond=None)[0]
            success = True
        except:
            delta_s = np.zeros_like(g)
            success = False

    return delta_s, success


# ============================================================================
# AR2-Simple: AR2 with Simple Update
# ============================================================================
# Relative paper: https://arxiv.org/pdf/2501.00404v2.pdf
# Procedure 1.2: Simple Update
# See the default parameters in the code.
# ----------------------------------------------------------------------------


def ar2_simple(
    model_func,
    model_grad,
    model_hess,
    s0,
    sigma0,
    eta1=0.01,
    eta2=0.95,
    gamma1=0.5,
    gamma2=3.0,
    sigma_min=1e-8,
    max_iterations=100,
    tol=1e-6,
    verbose=False,
):
    """
    AR2 algorithm with Simple Update strategy.

    This is designed to solve a subproblem by treating model_func as the objective.

    Parameters:
    - model_func: Function to minimize (e.g., AR3 model), signature: f(s) -> scalar
    - model_grad: Gradient of model, signature: grad(s) -> (n,)
    - model_hess: Hessian of model, signature: hess(s) -> (n, n)
    - s0: Initial point (n, 1) or (n,)
    - sigma0: Initial regularization parameter
    - eta1: Successful cutoff (default 0.01)
    - eta2: Very successful cutoff (default 0.95)
    - gamma1: Sigma decrease factor (default 0.5)
    - gamma2: Sigma increase factor (default 3.0)
    - sigma_min: Minimum sigma value (default 1e-8)
    - max_iterations: Maximum iterations (default 100)
    - tol: Convergence tolerance (default 1e-6)
    - verbose: Print iteration details (default False)

    Returns:
    - dict: {
        's_final': Final step,
        'iterations': Number of iterations,
        'converged': Boolean,
        'f_final': Final function value,
        'grad_norm_final': Final gradient norm
      }
    """
    s_k = s0.flatten().copy()
    sigma = max(sigma0, sigma_min)
    n = len(s_k)

    if verbose:
        print("AR2-Simple Inner Solver")
        print("-" * 60)

    for k in range(max_iterations):
        # Evaluate model at current point
        f_k = model_func(s_k)
        g_k = model_grad(s_k)
        H_k = model_hess(s_k)
        grad_norm = LA.norm(g_k)

        # Check convergence
        if grad_norm <= tol:
            # Check if Hessian is positive definite (local minimum, not saddle point)
            eigenvalues = LA.eigvalsh(H_k)
            min_eigenvalue = np.min(eigenvalues)
            min_eigenvalue_tol = 1e-6

            if min_eigenvalue >= min_eigenvalue_tol:
                # Positive definite Hessian -> local minimum
                if verbose:
                    print(
                        f"  AR2 converged at iteration {k}, ||∇m|| = {grad_norm:.6e}, min_eig = {min_eigenvalue:.6e}"
                    )
                return {
                    "s_final": s_k.reshape(-1, 1),
                    "iterations": k,
                    "converged": True,
                    "f_final": f_k,
                    "grad_norm_final": grad_norm,
                }
            else:
                # Negative eigenvalue -> saddle point, continue optimization
                # Use negative eigenvector direction to escape saddle point
                if verbose:
                    print(
                        f"  AR2 detected saddle point at iteration {k}, min_eig = {min_eigenvalue:.6e}, continuing..."
                    )

                eigenvals, eigenvecs = LA.eigh(H_k)
                neg_eigenvec_idx = np.where(eigenvals < -min_eigenvalue_tol)[0]

                if len(neg_eigenvec_idx) > 0:
                    # Use most negative eigenvector as escape direction
                    escape_dir = eigenvecs[:, neg_eigenvec_idx[0]]
                    # Small step in negative eigenvector direction
                    escape_step = 0.1 * escape_dir
                    s_k = s_k + escape_step
                    continue
                # If no clear negative eigenvector, continue with normal iteration

        if verbose and k % 10 == 0:
            print(
                f"  AR2 iter {k}: ||s|| = {LA.norm(s_k):.4e}, "
                f"m(s) = {f_k:.4e}, ||∇m|| = {grad_norm:.4e}, σ = {sigma:.4e}"
            )

        # Build AR2 model at s_k
        # The AR2 model is: m_AR2(delta_s) = m(s_k) + ∇m(s_k)'*delta_s +
        #                                      (1/2)*delta_s'*∇²m(s_k)*delta_s +
        #                                      (1/3)*sigma*||delta_s||^3
        # We want to minimize this over delta_s

        # For simplicity, use gradient descent with line search
        # or a single Newton-like step

        # Compute AR2 gradient and Hessian at s_k=0 (in the local frame)
        ar2_grad_0 = g_k  # ∇m(s_k)
        ar2_hess_0 = H_k  # ∇²m(s_k)

        # Try a Newton-like step first with sigma regularization
        # Solve: (H + sigma*||s_k||*I) * delta_s = -g_k
        norm_sk = LA.norm(s_k)
        reg_hess = H_k + sigma * norm_sk * np.eye(n)

        try:
            delta_s = LA.solve(reg_hess, -g_k)
        except LA.LinAlgError:
            # Use gradient descent if Newton fails
            delta_s = -g_k / (LA.norm(g_k) + 1e-10)

        # Ensure step is not too large
        norm_delta = LA.norm(delta_s)
        max_step = 1.0
        if norm_delta > max_step:
            delta_s = delta_s * (max_step / norm_delta)

        # Trial step
        s_trial = s_k + delta_s

        # Compute predicted decrease (using quadratic AR2 model)
        predicted_decrease = -(
            ar2_grad_0 @ delta_s
            + 0.5 * delta_s @ ar2_hess_0 @ delta_s
            + (1 / 3) * sigma * (LA.norm(delta_s) ** 3)
        )

        if predicted_decrease < 0:
            predicted_decrease = abs(predicted_decrease)

        # Compute actual decrease
        f_trial = model_func(s_trial)
        actual_decrease = f_k - f_trial

        # Compute ratio
        eps = np.finfo(float).eps * max(1, abs(f_k))
        if predicted_decrease > eps:
            rho = actual_decrease / predicted_decrease
        else:
            rho = -np.inf

        # Update based on rho
        if rho >= eta2:
            # Very successful
            s_k = s_trial
            sigma = max(gamma1 * sigma, sigma_min)
        elif rho >= eta1:
            # Successful
            s_k = s_trial
            sigma = max(sigma, sigma_min)
        else:
            # Unsuccessful - reject step
            sigma = gamma2 * sigma

    # Max iterations reached
    if verbose:
        print(f"  AR2 reached max iterations {max_iterations}")

    return {
        "s_final": s_k.reshape(-1, 1),
        "iterations": max_iterations,
        "converged": False,
        "f_final": model_func(s_k),
        "grad_norm_final": LA.norm(model_grad(s_k)),
    }


# ============================================================================
# AR2-Interp: AR2 with Interpolation Update
# ============================================================================
# Relative paper: https://arxiv.org/pdf/2501.00404v2.pdf
# Procedure 1.3: Interpolation Update
# See the default parameters in the code.
# ----------------------------------------------------------------------------


def padded_polyder_ar2(poly):
    """
    Compute polynomial derivative with zero-padding to maintain length.

    Parameters:
    - poly: Polynomial coefficients (highest degree first)

    Returns:
    - der: Derivative coefficients (padded to length len(poly)-1)

    Example:
    >>> poly = np.array([1, 2, 3])
    >>> padded_polyder_ar2(poly)
    array([3, 2])
    """
    der = np.polyder(poly)
    if len(der) < len(poly) - 1:
        der = np.concatenate([np.zeros(len(poly) - 1 - len(der)), der])
    return der


def real_roots_ar2(poly):
    """
    Find real roots of a polynomial.

    Parameters:
    - poly: Polynomial coefficients

    Returns:
    - roots: Array of real roots
    """
    if len(poly) == 0:
        return np.array([])
    roots_all = np.roots(poly)
    # Filter real roots (imaginary part is negligible)
    real_roots_arr = roots_all[np.abs(roots_all.imag) < 1e-10].real
    return real_roots_arr


def pos_boundary_points_ar2(poly_constraints):
    # ----------------------------------------------------------------------------
    # Relative paper: https://arxiv.org/pdf/2501.00404v2.pdf
    # Page 19, Equation (14) - (18)
    # ----------------------------------------------------------------------------
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
        points = real_roots_ar2(poly_constraints[i])
        points = points[points > 0]

        # Check if points satisfy all other constraints
        for j in list(range(i)) + list(range(i + 1, num_constraints)):
            if len(points) > 0:
                vals = np.polyval(poly_constraints[j], points)
                points = points[vals <= 10 * np.finfo(float).eps]

        if len(points) > 0:
            boundary.extend(points.tolist())

    return np.array(boundary)


def optimize_sigma_ar2(
    taylor_poly, prev_sigma, norm_step, constraint_poly, decrease_sigma
):
    # ----------------------------------------------------------------------------
    # Relative paper: https://arxiv.org/pdf/2501.00404v2.pdf
    # Procedure 3.2 - Interpolation Update
    # Page 19, Equation (14) - (18)
    # ----------------------------------------------------------------------------
    """
    Find optimal sigma satisfying interpolation constraints for AR2.

    Parameters:
    - taylor_poly: Taylor polynomial coefficients [c2, c1, c0]
    - prev_sigma: Previous sigma value
    - norm_step: Norm of the step
    - constraint_poly: Constraint polynomial
    - decrease_sigma: If True, find largest sigma <= prev_sigma; else smallest >= prev_sigma

    Returns:
    - sigma: Optimal sigma value (or nan if not found)
    - alpha: Corresponding alpha value (or nan)
    """
    p = len(taylor_poly) - 1  # p=2 for AR2
    taylor_der_poly = padded_polyder_ar2(taylor_poly)

    # Compute sigma as a function of alpha
    def compute_sigma(alpha):
        with np.errstate(divide="ignore", invalid="ignore"):
            result = -np.polyval(taylor_der_poly, alpha) / (
                norm_step ** (p + 1) * alpha**p
            )
            return np.where(np.isfinite(result), result, 0)

    # Build constraint polynomials
    sign = -1 if decrease_sigma else 1
    prev_sigma_poly = sign * np.concatenate(
        [[prev_sigma * norm_step ** (p + 1)], taylor_der_poly]
    )

    # For AR2, no pre-rejection needed, use standard constraints
    local_min_poly_temp = -padded_polyder_ar2(taylor_der_poly) + p * taylor_der_poly
    local_min_poly = np.concatenate([local_min_poly_temp, [0]])

    # Collect all constraints
    constraints = [constraint_poly, prev_sigma_poly, taylor_der_poly, local_min_poly]

    # Find boundary points
    alpha_options = pos_boundary_points_ar2(constraints)

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


def construct_taylor_poly_ar2(s, f, g, H):
    """
    Construct 1D Taylor polynomial along step direction s for AR2.

    Returns polynomial t(alpha) = f(x + alpha*s) (Taylor approximation)
    Coefficients are [c2, c1, c0] for c2*alpha^2 + c1*alpha + c0

    Parameters:
    - s: Step direction (n,) or (n, 1)
    - f: Function value
    - g: Gradient (n,) or (n, 1)
    - H: Hessian (n, n)

    Returns:
    - taylor_poly: Polynomial coefficients [c2, c1, c0]
    """
    s = s.flatten()
    g = g.flatten()

    # Second-order term
    c2 = 0.5 * s @ H @ s

    # Linear term
    c1 = g @ s

    # Constant term
    c0 = f

    return np.array([c2, c1, c0])


def solve_ar2_subproblem(g, H, sigma, f=0, method="trust-constr", max_iter=1000):
    """
    Solve the AR2 subproblem using scipy.

    minimize m(s) = f + g'*s + (1/2)*s'*H*s + (1/3)*sigma*||s||^3

    Parameters:
    - g: Gradient (n, 1) or (n,)
    - H: Hessian (n, n)
    - sigma: Regularization parameter
    - f: Function value (default: 0)
    - method: Optimization method
    - max_iter: Maximum iterations

    Returns:
    - s: Step vector (n, 1)
    - success: Whether solver succeeded
    - n_iter: Number of iterations
    """
    n = len(g) if len(g.shape) == 1 else g.shape[0]
    g = g.flatten()

    # Define objective and its derivatives
    def objective(s):
        return ar2_model(s, f, g, H, sigma)

    def gradient(s):
        return ar2_model_gradient(s, g, H, sigma)

    def hessian(s):
        return ar2_model_hessian(s, H, sigma)

    # Initial guess
    s0 = np.zeros(n)

    # Solve using scipy
    try:
        if method == "trust-constr":
            # Remark-Yubo: We employs the scipy.optimize package for optimization. 
            # In contrast, the original paper utilized the MCM to locate the global minimum.
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

    except Exception:
        s = np.zeros((n, 1))
        success = False
        n_iter = 0

    return s, success, n_iter


def ar2_interp(
    fx,
    dx,
    d2x,
    x0,
    max_iterations,
    tol,
    param_list=None,
    verbose=True,
):
    """
    AR2-Interp: AR2 with Interpolation Update.

    This is the best-performing AR2 variant from the paper. It uses interpolation
    to adaptively update sigma, but does NOT use pre-rejection (as AR2 global
    minimizers are always persistent).

    Parameters:
    - fx: Objective function f(x): R^n -> R
    - dx: Gradient function ∇f(x): R^n -> R^n
    - d2x: Hessian function ∇²f(x): R^(n×n)
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
    interp_updates = 0

    # History tracking
    x_history = [x0.copy()]
    s_history = []
    f_history = [fx(x0)]
    grad_norm_history = [LA.norm(g0)]
    sigma_history = [sigma]
    rho_history = []
    success_history = []

    if verbose:
        print("=" * 100)
        print("AR2-Interp (AR2 with Interpolation Update)")
        print("=" * 100)
        print(f"Parameters: eta1={eta1}, eta2={eta2}, gamma1={gamma1}, gamma3={gamma3}")
        print(f"            beta={beta}, alpha_max={alpha_max}, sigma_min={sigma_min}")
        print(f"Initial sigma: {sigma}")
        print()

    while k < max_iterations:
        # Step 1: Compute derivatives
        f_k = fx(x_k)
        g_k = dx(x_k)
        H_k = d2x(x_k)
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
        s_k, sub_success, sub_iter = solve_ar2_subproblem(g_k, H_k, sigma, f_k)

        subproblem_counter += 1
        norm_s = LA.norm(s_k)

        if verbose:
            print(f"  ||s_k||: {norm_s:.6e}")

        # Step 4: Construct Taylor polynomial along step direction
        taylor_poly = construct_taylor_poly_ar2(s_k, f_k, g_k, H_k)

        # Step 5: Compute model value
        model_poly_0 = np.polyval(taylor_poly, 0)
        model_poly_1 = np.polyval(taylor_poly, 1)

        # Predicted decrease (using Taylor approximation)
        predicted_decrease = f_k - model_poly_1

        if predicted_decrease < 0:
            predicted_decrease = abs(predicted_decrease)

        # Step 6: Compute actual decrease
        f_trial = fx(x_k + s_k)
        actual_decrease = f_k - f_trial

        # Compute acceptance ratio
        eps_val = np.finfo(float).eps * max(1, abs(f_k))
        if predicted_decrease > eps_val:
            rho = actual_decrease / predicted_decrease
        else:
            rho = -np.inf

        rho_history.append(rho)

        if verbose:
            print(f"  Predicted decrease: {predicted_decrease:.6e}")
            print(f"  Actual decrease: {actual_decrease:.6e}")
            print(f"  rho: {rho:.6e}")

        # Step 7: Update iterate and sigma using Interpolation Update
        success = False

        if rho >= 1 + np.finfo(float).eps:
            # Extremely successful: use interpolation to decrease sigma
            # Construct interpolation polynomial
            f_plus = f_k - actual_decrease
            interp_poly = np.concatenate([[f_plus - model_poly_1], taylor_poly])

            # Compute constraint for successful_sigma
            taylor_der_poly = padded_polyder_ar2(taylor_poly)
            p = len(taylor_poly) - 1  # p=2
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
                new_sigma, alpha_opt = optimize_sigma_ar2(
                    taylor_poly, sigma, norm_s, constraint_poly, decrease_sigma=True
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

        elif rho >= eta2:
            # Very successful
            sigma = gamma1 * sigma
            sigma = max(sigma, sigma_min)
            x_k = x_k + s_k
            success = True

            if verbose:
                print(f"  Very successful! Accepting step, sigma={sigma:.6e}")

        elif rho >= eta1:
            # Successful
            sigma = max(sigma, sigma_min)
            x_k = x_k + s_k
            success = True

            if verbose:
                print(f"  Successful! Accepting step, sigma={sigma:.6e}")

        elif rho >= -np.finfo(float).eps:
            # Unsuccessful
            sigma = 3.0 * sigma

            if verbose:
                print(
                    f"  Unsuccessful. Rejecting step, increasing sigma to {sigma:.6e}"
                )

        else:
            # Extremely unsuccessful: use interpolation to increase sigma
            f_plus = f_k - actual_decrease
            interp_poly = np.concatenate([[f_plus - model_poly_1], taylor_poly])

            # Build constraint for unsuccessful_sigma
            # actual_decrease_poly: [-interp_poly[:-1], 0]
            actual_decrease_poly = np.concatenate([-interp_poly[:-1], [0]])
            predicted_decrease_poly = np.concatenate([[0], -taylor_poly[:-1], [0]])
            constraint_poly = actual_decrease_poly - eta1 * predicted_decrease_poly
            constraint_poly = constraint_poly[:-1]  # Remove constant term

            new_sigma, _ = optimize_sigma_ar2(
                taylor_poly, sigma, norm_s, -constraint_poly, decrease_sigma=False
            )

            if not np.isnan(new_sigma) and new_sigma >= 3.0 * sigma:
                sigma = min(new_sigma, 100 * sigma)  # Cap at max_increase
                interp_updates += 1
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
        print("AR2-Interp Summary:")
        print(f"  Converged: {k < max_iterations}")
        print(f"  Iterations: {k}")
        print(f"  Successful iterations: {n_successful}")
        print(f"  Unsuccessful iterations: {n_unsuccessful}")
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
        "subproblem_solves": subproblem_counter,
        "interp_updates": interp_updates,
        "n_successful": n_successful,
        "n_unsuccessful": n_unsuccessful,
        "converged": k < max_iterations,
    }
