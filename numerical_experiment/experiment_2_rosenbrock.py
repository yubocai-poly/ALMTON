"""
Experiment 2: High-Dimensional Rosenbrock Stress Test

Purpose: Test algorithms on Rosenbrock-n function with narrow valleys.
Rosenbrock function is known for its narrow parabolic valley, making it
an excellent test case for algorithms that can utilize higher-order curvature
information to escape "flat" regions.

Mathematical Definition (Rosenbrock-n):
f(x) = sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

Experiment Setup:
- Dimensions: N = 5, 20, 50, 100
- Starting points:
  * Standard difficult point: x0 = [-1.2, 1.0, ..., 1.0]
  * Random points: 10 random seeds, x0 ~ U(-2, 2)
- Comparison algorithms:
  * ALMTON
  * L-BFGS (scipy.optimize)
  * Newton-CG (scipy.optimize)
  * AR3-Interp

Metrics:
- Iterations
- Function evaluations
- Wall-clock time
- Stagnation escape behavior
"""

import numpy as np
import numpy.linalg as LA
import sys
import time
import pandas as pd
import os
import pickle
from scipy.optimize import minimize, fmin_l_bfgs_b
from functools import partial
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# Add path
try:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(base_dir, "src")
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
except Exception as e:
    base_dir = os.getcwd()
    src_dir = os.path.join(base_dir, "src")
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

try:
    from AdaptiveFramework import almton_simple, almton_heuristic
    from AR3 import ar3_interp
    from NewtonFunctions import init_func, init_params
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"sys.path: {sys.path[:5]}")
    raise


# ============================================================================
# Algorithm Wrappers
# ============================================================================


def run_lbfgs(fx, dx, x0, max_iterations, tol):
    """
    Run L-BFGS-B algorithm using scipy.optimize.fmin_l_bfgs_b

    Returns:
    - x_final: Final point
    - converged: Boolean
    - iterations: Number of iterations
    - nfev: Number of function evaluations
    - time_elapsed: Wall-clock time
    """
    x0_flat = x0.flatten()
    start_time = time.time()

    # Track function evaluations
    nfev_counter = {"count": 0}

    def func_wrapper(x):
        nfev_counter["count"] += 1
        return fx(x.reshape(-1, 1))

    def grad_wrapper(x):
        return dx(x.reshape(-1, 1)).flatten()

    try:
        result = fmin_l_bfgs_b(
            func_wrapper,
            x0_flat,
            fprime=grad_wrapper,
            maxiter=max_iterations,
            factr=tol * 1e12,  # factr = (f_k - f_{k+1})/max(|f_k|,|f_{k+1}|,1) * factr
            pgtol=tol,
            iprint=-1,  # Suppress output
        )

        x_final = result[0].reshape(-1, 1)
        converged = result[2]["warnflag"] == 0
        iterations = result[2]["nit"]
        nfev = nfev_counter["count"]
        time_elapsed = time.time() - start_time

    except Exception as e:
        x_final = x0
        converged = False
        iterations = max_iterations
        nfev = nfev_counter["count"]
        time_elapsed = time.time() - start_time

    return x_final, converged, iterations, nfev, time_elapsed


def run_newton_cg(fx, dx, d2x, x0, max_iterations, tol):
    """
    Run Newton-CG algorithm using scipy.optimize.minimize

    Returns:
    - x_final: Final point
    - converged: Boolean
    - iterations: Number of iterations
    - nfev: Number of function evaluations
    - time_elapsed: Wall-clock time
    """
    x0_flat = x0.flatten()
    start_time = time.time()

    # Track function evaluations
    nfev_counter = {"count": 0}
    njev_counter = {"count": 0}

    def func_wrapper(x):
        nfev_counter["count"] += 1
        return fx(x.reshape(-1, 1))

    def grad_wrapper(x):
        njev_counter["count"] += 1
        return dx(x.reshape(-1, 1)).flatten()

    def hess_wrapper(x):
        return d2x(x.reshape(-1, 1))

    try:
        result = minimize(
            func_wrapper,
            x0_flat,
            method="Newton-CG",
            jac=grad_wrapper,
            hess=hess_wrapper,
            options={"maxiter": max_iterations, "xtol": tol, "disp": False},
        )

        x_final = result.x.reshape(-1, 1)
        converged = result.success
        iterations = result.nit
        nfev = nfev_counter["count"]
        time_elapsed = time.time() - start_time

    except Exception as e:
        x_final = x0
        converged = False
        iterations = max_iterations
        nfev = nfev_counter["count"]
        time_elapsed = time.time() - start_time

    return x_final, converged, iterations, nfev, time_elapsed


# ============================================================================
# Main Experiment Function
# ============================================================================


def run_single_algorithm_rosenbrock(args):
    """
    Run a single algorithm on Rosenbrock function.

    Parameters:
    - args: (x0, func_name, algorithm_name, max_iterations, tol, param_dict, sdp_tol)

    Returns:
    - dict: Result dictionary
    """
    x0, func_name, algorithm_name, max_iterations, tol, param_dict, sdp_tol = args

    [fX, fx, dx, d2x, d3x] = init_func(func_name)
    x0 = np.array(x0).reshape(-1, 1)

    result = {
        "algorithm": algorithm_name,
        "x0": x0.flatten(),
        "converged": False,
        "iterations": max_iterations,
        "nfev": 0,
        "final_f": np.inf,
        "final_grad_norm": np.inf,
        "time_elapsed": 0.0,
        "failure_reason": None,
    }

    try:
        if algorithm_name == "L-BFGS":
            x_final, converged, iters, nfev, time_elapsed = run_lbfgs(
                fx, dx, x0, max_iterations, tol
            )
            result["iterations"] = iters
            result["nfev"] = nfev
            result["time_elapsed"] = time_elapsed
            result["converged"] = converged
            if converged:
                result["final_f"] = fx(x_final)
                result["final_grad_norm"] = LA.norm(dx(x_final))
            else:
                result["failure_reason"] = "max_iterations_exceeded"

        elif algorithm_name == "Newton-CG":
            x_final, converged, iters, nfev, time_elapsed = run_newton_cg(
                fx, dx, d2x, x0, max_iterations, tol
            )
            result["iterations"] = iters
            result["nfev"] = nfev
            result["time_elapsed"] = time_elapsed
            result["converged"] = converged
            if converged:
                result["final_f"] = fx(x_final)
                result["final_grad_norm"] = LA.norm(dx(x_final))
            else:
                result["failure_reason"] = "max_iterations_exceeded"

        elif algorithm_name == "AR3-Interp":
            ar3_interp_params = param_dict.get(
                "ar3_interp", [0.01, 0.95, 0.5, 0.1, 0.01, 2.0, 1e-8]
            )
            start_time = time.time()
            r = ar3_interp(
                fx,
                dx,
                d2x,
                d3x,
                x0,
                max_iterations,
                tol,
                ar3_interp_params,
                verbose=False,
            )
            time_elapsed = time.time() - start_time

            result["iterations"] = r["iterations"]
            # Function evaluations: each iteration evaluates f, dx, d2x, d3x once
            # Plus subproblem evaluations (approximate)
            result["nfev"] = r["iterations"] * 4 + r.get("subproblem_solves", 0) * 10
            result["time_elapsed"] = time_elapsed
            result["converged"] = r["converged"]
            if r["converged"]:
                result["final_f"] = r["f_history"][-1]
                result["final_grad_norm"] = r["grad_norm_history"][-1]
            else:
                if r["iterations"] >= max_iterations:
                    result["failure_reason"] = "max_iterations_exceeded"
                else:
                    result["failure_reason"] = "subproblem_failure"

        elif algorithm_name == "ALMTON":
            almton_params = param_dict.get("almton", [0.1, 0.01, 0.1, 2.0])
            start_time = time.time()
            r = almton_simple(
                fx,
                dx,
                d2x,
                d3x,
                x0,
                max_iterations,
                tol,
                almton_params,
                verbose=False,
                sdp_tol=sdp_tol,
            )
            time_elapsed = time.time() - start_time

            result["iterations"] = r["iterations"]
            # Function evaluations: each iteration evaluates f, dx, d2x, d3x once
            # Plus SDP solves (each SDP solve doesn't require function evaluations)
            result["nfev"] = r["iterations"] * 4 + r.get("sdp_solves", 0)
            result["time_elapsed"] = time_elapsed
            result["converged"] = r["converged"]
            if r["converged"]:
                result["final_f"] = r["f_history"][-1]
                result["final_grad_norm"] = r["grad_norm_history"][-1]
            else:
                if r.get("sigma_exceeded", False):
                    result["failure_reason"] = "sigma_exceeded"
                elif r["iterations"] >= max_iterations:
                    result["failure_reason"] = "max_iterations_exceeded"
                else:
                    result["failure_reason"] = "sdp_solver_failure"

        elif algorithm_name == "almton_heuristic":
            almton_heuristic_params = param_dict.get(
                "almton_heuristic", [0.1, 0.01, 0.1, 2.0]
            )
            start_time = time.time()
            r = almton_heuristic(
                fx,
                dx,
                d2x,
                d3x,
                x0,
                max_iterations,
                tol,
                almton_heuristic_params,
                verbose=False,
                sdp_tol=sdp_tol,
            )
            time_elapsed = time.time() - start_time

            result["iterations"] = r["iterations"]
            # Function evaluations: each iteration evaluates f, dx, d2x, d3x once
            # Plus SDP solves
            result["nfev"] = r["iterations"] * 4 + r.get("sdp_solves", 0)
            result["time_elapsed"] = time_elapsed
            result["converged"] = r["converged"]
            if r["converged"]:
                result["final_f"] = r["f_history"][-1]
                result["final_grad_norm"] = r["grad_norm_history"][-1]
            else:
                if r.get("sigma_exceeded", False):
                    result["failure_reason"] = "sigma_exceeded"
                elif r["iterations"] >= max_iterations:
                    result["failure_reason"] = "max_iterations_exceeded"
                else:
                    result["failure_reason"] = "sdp_solver_failure"

    except Exception as e:
        result["error"] = str(e)
        result["converged"] = False
        result["failure_reason"] = "exception"

    return result


def experiment_2_rosenbrock(
    n_dim=20,
    n_random_trials=10,
    max_iterations=1000,
    tol=1e-6,
    save_path=None,
    use_parallel=True,
    n_jobs=None,
    sdp_tol=1e-3,
):
    """
    Experiment 2: High-Dimensional Rosenbrock Stress Test

    Parameters:
    - n_dim: Dimension of Rosenbrock function (default: 20)
    - n_random_trials: Number of random starting points (default: 10)
    - max_iterations: Maximum iterations (default: 1000)
    - tol: Convergence tolerance (default: 1e-6)
    - save_path: Path to save results (default: results/experiment_2/)
    - use_parallel: Whether to use parallel computation (default: True)
    - n_jobs: Number of parallel processes (default: None, uses cpu_count())
    - sdp_tol: Fixed SDP tolerance for ALMTON algorithms (default: 1e-3)
               Set to None to use dynamic tolerance based on gradient norm

    Returns:
    - dict: Results dictionary
    """
    print("=" * 120)
    print(f"Experiment 2: High-Dimensional Rosenbrock Stress Test (N={n_dim})")
    print("=" * 120)

    func_name = f"Rosenbrock-{n_dim}"

    # Initialize function
    [fX, fx, dx, d2x, d3x] = init_func(func_name)
    [XMIN, XMAX, YMIN, YMAX, x_min] = init_params(func_name)

    # Algorithm list
    algorithms = ["L-BFGS", "Newton-CG", "AR3-Interp", "ALMTON", "almton_heuristic"]

    # Parameter settings
    param_dict = {
        "ar3_interp": [0.01, 0.95, 0.5, 0.1, 0.01, 2.0, 1e-8],
        "almton": [0.1, 0.01, 0.1, 2.0],
        "almton_heuristic": [0.1, 0.01, 0.1, 2.0],
    }

    # Generate starting points
    start_points = []

    # Standard difficult point: [-1.2, 1.0, ..., 1.0]
    x0_standard = np.zeros(n_dim)
    x0_standard[0] = -1.2
    x0_standard[1:] = 1.0
    start_points.append(("standard", x0_standard))

    # Random points
    np.random.seed(42)  # For reproducibility
    for i in range(n_random_trials):
        x0_random = np.random.uniform(-2, 2, n_dim)
        start_points.append((f"random_{i+1}", x0_random))

    print(f"\nGenerated {len(start_points)} starting points:")
    print(f"  - 1 standard difficult point: [-1.2, 1.0, ..., 1.0]")
    print(f"  - {n_random_trials} random points: U(-2, 2)")

    # Determine number of parallel processes
    if n_jobs is None:
        n_jobs = cpu_count()

    # Run all algorithms
    all_results = {}

    for algo_name in algorithms:
        print(f"\nTesting algorithm: {algo_name}")
        print("-" * 120)

        # Prepare arguments for parallel execution
        args_list = [
            (x0, func_name, algo_name, max_iterations, tol, param_dict, sdp_tol)
            for point_name, x0 in start_points
        ]

        # Run in parallel or sequentially
        if use_parallel and len(start_points) > 1:
            with Pool(processes=n_jobs) as pool:
                results = pool.map(run_single_algorithm_rosenbrock, args_list)
        else:
            results = []
            for args in args_list:
                results.append(run_single_algorithm_rosenbrock(args))

        # Add point names to results
        for i, (point_name, _) in enumerate(start_points):
            results[i]["point_name"] = point_name

        # Print results
        for result in results:
            if result["converged"]:
                print(
                    f"  {result['point_name']:15s}: converged in {result['iterations']:4d} iter, "
                    f"{result['nfev']:5d} fev, {result['time_elapsed']:6.3f}s, "
                    f"f={result['final_f']:.6e}"
                )
            else:
                print(
                    f"  {result['point_name']:15s}: failed ({result.get('failure_reason', 'unknown')})"
                )

        all_results[algo_name] = results
        print(f"  Completed: {algo_name}")

    # Compute statistics
    print("\n" + "=" * 120)
    print("Statistical Results (only converged points)")
    print("=" * 120)

    stats = {}
    detailed_data = []

    for algo_name in algorithms:
        results = all_results[algo_name]

        # Only count successfully converged points
        converged_results = [r for r in results if r["converged"]]
        n_converged = len(converged_results)
        n_total = len(results)
        conv_rate = n_converged / n_total if n_total > 0 else 0

        if n_converged > 0:
            iterations = [r["iterations"] for r in converged_results]
            nfevs = [r["nfev"] for r in converged_results]
            times = [r["time_elapsed"] for r in converged_results]
            final_f = [r["final_f"] for r in converged_results]

            median_iters = np.median(iterations)
            q1_iters = np.percentile(iterations, 25)
            q3_iters = np.percentile(iterations, 75)
            iqr_iters = q3_iters - q1_iters

            median_nfev = np.median(nfevs)
            q1_nfev = np.percentile(nfevs, 25)
            q3_nfev = np.percentile(nfevs, 75)
            iqr_nfev = q3_nfev - q1_nfev

            median_time = np.median(times)
            q1_time = np.percentile(times, 25)
            q3_time = np.percentile(times, 75)
            iqr_time = q3_time - q1_time

            median_final_f = np.median(final_f)
        else:
            median_iters = np.nan
            q1_iters = np.nan
            q3_iters = np.nan
            iqr_iters = np.nan
            median_nfev = np.nan
            q1_nfev = np.nan
            q3_nfev = np.nan
            iqr_nfev = np.nan
            median_time = np.nan
            q1_time = np.nan
            q3_time = np.nan
            iqr_time = np.nan
            median_final_f = np.nan

        # Failure reason statistics
        failure_reasons = {}
        failed_results = [r for r in results if not r["converged"]]
        for r in failed_results:
            reason = r.get("failure_reason", "unknown")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        stats[algo_name] = {
            "median_iters": median_iters,
            "q1_iters": q1_iters,
            "q3_iters": q3_iters,
            "iqr_iters": iqr_iters,
            "median_nfev": median_nfev,
            "q1_nfev": q1_nfev,
            "q3_nfev": q3_nfev,
            "iqr_nfev": iqr_nfev,
            "median_time": median_time,
            "q1_time": q1_time,
            "q3_time": q3_time,
            "iqr_time": iqr_time,
            "conv_rate": conv_rate,
            "median_final_f": median_final_f,
            "n_converged": n_converged,
            "n_total": n_total,
            "failure_reasons": failure_reasons,
        }

        # Detailed data
        for result in results:
            detailed_data.append(
                {
                    "Algorithm": algo_name,
                    "Point Name": result["point_name"],
                    "Converged": result["converged"],
                    "Iterations": (
                        result["iterations"] if result["converged"] else np.nan
                    ),
                    "Function Evals": result["nfev"] if result["converged"] else np.nan,
                    "Time (s)": result["time_elapsed"],
                    "Final f(x)": (
                        result["final_f"] if result["converged"] else np.nan
                    ),
                    "Final ||∇f||": (
                        result["final_grad_norm"] if result["converged"] else np.nan
                    ),
                    "Failure Reason": result.get("failure_reason", ""),
                }
            )

    # Print statistics table
    print(
        f"\n{'Algorithm':<15} {'Median Iters (IQR)':<25} {'Median Fevals (IQR)':<25} {'Median Time(s) (IQR)':<25} "
        f"{'Conv Rate':<12} {'Median Final f':<15}"
    )
    print("-" * 120)

    for algo_name in algorithms:
        s = stats[algo_name]
        if not np.isnan(s["median_iters"]):
            iters_str = (
                f"{s['median_iters']:.1f} ({s['q1_iters']:.1f}-{s['q3_iters']:.1f})"
            )
            nfev_str = f"{s['median_nfev']:.1f} ({s['q1_nfev']:.1f}-{s['q3_nfev']:.1f})"
            time_str = f"{s['median_time']:.3f} ({s['q1_time']:.3f}-{s['q3_time']:.3f})"
            print(
                f"{algo_name:<15} {iters_str:<25} {nfev_str:<25} {time_str:<25} "
                f"{s['conv_rate']:>6.1%}       {s['median_final_f']:>12.6e}"
            )
        else:
            print(
                f"{algo_name:<15} {'N/A':<25} {'N/A':<25} {'N/A':<25} "
                f"{s['conv_rate']:>6.1%}       {'N/A':<15}"
            )
        # Print failure reasons
        if s["failure_reasons"]:
            print(f"  Failure reasons:")
            for reason, count in sorted(
                s["failure_reasons"].items(), key=lambda x: -x[1]
            ):
                print(
                    f"    {reason}: {count} ({count/(s['n_total']-s['n_converged'])*100:.1f}%)"
                )

    # Save results
    if save_path:
        os.makedirs(save_path, exist_ok=True)

        # Create DataFrames
        df_detailed = pd.DataFrame(detailed_data)
        df_summary = pd.DataFrame(
            [
                {
                    "Algorithm": algo_name,
                    "Median Iters": stats[algo_name]["median_iters"],
                    "Q1 Iters": stats[algo_name]["q1_iters"],
                    "Q3 Iters": stats[algo_name]["q3_iters"],
                    "IQR Iters": stats[algo_name]["iqr_iters"],
                    "Median Fevals": stats[algo_name]["median_nfev"],
                    "Q1 Fevals": stats[algo_name]["q1_nfev"],
                    "Q3 Fevals": stats[algo_name]["q3_nfev"],
                    "IQR Fevals": stats[algo_name]["iqr_nfev"],
                    "Median Time(s)": stats[algo_name]["median_time"],
                    "Q1 Time(s)": stats[algo_name]["q1_time"],
                    "Q3 Time(s)": stats[algo_name]["q3_time"],
                    "IQR Time(s)": stats[algo_name]["iqr_time"],
                    "Conv Rate": stats[algo_name]["conv_rate"],
                    "Median Final f": stats[algo_name]["median_final_f"],
                    "N Converged": stats[algo_name]["n_converged"],
                    "N Total": stats[algo_name]["n_total"],
                    "Failure Reasons": (
                        "; ".join(
                            [
                                f"{k}:{v}"
                                for k, v in sorted(
                                    stats[algo_name]["failure_reasons"].items(),
                                    key=lambda x: -x[1],
                                )
                            ]
                        )
                        if stats[algo_name]["failure_reasons"]
                        else ""
                    ),
                }
                for algo_name in algorithms
            ]
        )

        # Save to Excel
        excel_path = os.path.join(save_path, f"Rosenbrock-{n_dim}_experiment_2.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
            df_detailed.to_excel(writer, sheet_name="Detailed", index=False)

        print(f"\nResults saved to: {excel_path}")

        # Save pickle
        pickle_path = os.path.join(
            save_path, f"Rosenbrock-{n_dim}_experiment_2_data.pkl"
        )
        pickle_data = {
            "all_results": all_results,
            "stats": stats,
            "detailed_data": detailed_data,
            "n_dim": n_dim,
            "n_random_trials": n_random_trials,
            "max_iterations": max_iterations,
            "tol": tol,
        }
        with open(pickle_path, "wb") as f:
            pickle.dump(pickle_data, f)
        print(f"Detailed data saved to: {pickle_path}")

    return {
        "all_results": all_results,
        "stats": stats,
        "detailed_data": detailed_data,
    }


# ============================================================================
# Main Function
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Experiment 2: Rosenbrock Stress Test"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=20,
        choices=[5, 20, 50, 100],
        help="Dimension of Rosenbrock function (default: 20)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of random starting points (default: 10)",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=1000,
        help="Maximum iterations (default: 1000)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance (default: 1e-6)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Save path (default: results/experiment_2/)",
    )
    parser.add_argument(
        "--no_parallel",
        action="store_true",
        help="Disable parallel computation",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Number of parallel processes (default: cpu_count())",
    )
    parser.add_argument(
        "--sdp_tol",
        type=float,
        default=1e-3,
        help="Fixed SDP tolerance for ALMTON algorithms (default: 1e-3). Set to None to use dynamic tolerance",
    )

    args = parser.parse_args()

    # Set save path
    if args.save_path is None:
        args.save_path = os.path.join(
            os.path.dirname(__file__), "..", "results", "experiment_2"
        )

    # Handle sdp_tol: if string "None", convert to None
    sdp_tol_value = None if str(args.sdp_tol).lower() == "none" else args.sdp_tol

    # Run experiment
    result = experiment_2_rosenbrock(
        n_dim=args.dim,
        n_random_trials=args.n_trials,
        max_iterations=args.max_iterations,
        tol=args.tol,
        save_path=args.save_path,
        use_parallel=not args.no_parallel,
        n_jobs=args.n_jobs,
        sdp_tol=sdp_tol_value,
    )

    print("\n" + "=" * 120)
    print("Experiment 2 completed!")
    print("=" * 120)
