# Adaptive Levenberg-Marquardt Third-Order Newton Method (ALMTON)

This repository is the source code for the algorithm ```ALMTON``` (**Adaptive Levenberg-Marquardt Third-Order Newton Method**) proposed in our paper:

> Yubo Cai, Wenqi Zhu, Coralia Cartis, and Gioele Zardini, "A Globally Convergent Third-Order Newton Method via Unified Semidefinite Programming Subproblems," 2026. [arXiv:2603.09682](https://arxiv.org/abs/2603.09682).

Part of our codebase for solving the SDP subproblems builds upon the implementation from [Third_Order_Newton](https://github.com/jeffreyzhang92/Third_Order_Newton), which accompanies the paper:

> Olha Silina and Jeffrey Zhang, "An Unregularized Third Order Newton Method," 2023. [arXiv:2209.10051](https://arxiv.org/abs/2209.10051).

The implementations of AR2 and AR3 in this repository are inspired by the MATLAB code in [ar3-matlab](https://github.com/karlwelzel/ar3-matlab), which accompanies the paper:

> C. Cartis, R. A. Hauser, Y. Liu, K. Welzel, and W. Zhu, "Efficient Implementation of Third-order Tensor Methods with Adaptive Regularization for Unconstrained Optimization," 2025. https://arxiv.org/abs/2501.00404

Since the original repository is implemented in MATLAB, we developed a Python reimplementation of the algorithmic framework to enable direct comparisons with our ALMTON implementation.

The main components of this repository are organized as follows:

- [src/UnregularizedThirdOrder.py](src/UnregularizedThirdOrder.py) implements the SDP subproblem solver used in ALMTON.
- [src/NewtonFunctions.py](src/NewtonFunctions.py) defines the test problems, including the objective functions, gradients, Hessians, and third-order derivatives.
- [src/AdaptiveFramework.py](src/AdaptiveFramework.py) contains the adaptive framework, including the ALMTON implementation and parameter update rules.
- [src/AR2.py](src/AR2.py) and [src/AR3.py](src/AR3.py) implement the AR2 and AR3 algorithms, respectively.
- All experimental scripts are located in [numerical_experiment](numerical_experiment), and the generated results are stored in [results](results).

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### SDP Solvers

Our algorithm solves semidefinite programming (SDP) subproblems and supports the following solvers:

- **[MOSEK](https://www.mosek.com/)** (recommended): A commercial solver that provides an [academic license](https://www.mosek.com/products/academic-licenses/) free of charge. After obtaining a license, install via:
  ```bash
  pip install mosek
  ```
- **[SCS](https://www.cvxgrp.org/scs/)**: An open-source solver, installed automatically with CVXPY.
- **[CVXOPT](https://cvxopt.org/)**: An open-source solver, install via:
  ```bash
  pip install cvxopt
  ```

By default, the solver selection is set to `"auto"`, which prioritizes MOSEK if available and falls back to SCS.

### Running the Numerical Experiments

All numerical experiments used in the paper are located in [numerical_experiment](numerical_experiment). We recommend running the scripts from the repository root:

```bash
python3 numerical_experiment/<script_name>.py ...
```

Each script exposes a command-line interface, so the quickest way to inspect the available options is:

```bash
python3 numerical_experiment/<script_name>.py --help
```

The most commonly adjusted runtime options are:

- `--max_iterations`: maximum number of iterations.
- `--tol`: stopping tolerance.
- `--save_path`: output directory for figures, tables, and serialized data.
- `--no_parallel`: disable multiprocessing.
- `--n_jobs`: number of worker processes when parallel execution is enabled.

#### Experiment 1: Two-Dimensional Benchmark Tests

The script [numerical_experiment/experiment_1.py](numerical_experiment/experiment_1.py) covers Experiments 1.1, 1.2, and 1.3.

Typical usage:

```bash
python3 numerical_experiment/experiment_1.py --experiment 1.1 --function Himmelblau
python3 numerical_experiment/experiment_1.py --experiment 1.2 --function Beale
python3 numerical_experiment/experiment_1.py --experiment 1.3
python3 numerical_experiment/experiment_1.py --experiment all --function all
```

Main options:

- `--experiment {1.1,1.2,1.3,both,all}`:
  `1.1` runs the dense-grid test, `1.2` runs the tabular comparison, `1.3` generates Dolan-More performance profiles, `both` runs `1.1` and `1.2`, and `all` runs `1.1`, `1.2`, and `1.3`.
- `--function <name>`:
  benchmark function name. The default is `Himmelblau`. Setting `--function all` runs the default collection used by the performance-profile pipeline.
- `--trials <int>`:
  number of trials for Experiment 1.2 when random sampling is used.
- `--grid_size <nx> <ny>`:
  grid size for Experiment 1.1. The default is `20 20`.
- `--functions_13 <f1> <f2> ...`:
  functions included in Experiment 1.3. By default, the script uses `Beale`, `Himmelblau`, `McCormick`, and `Bohachevsky`.
- `--grid_size_13 <nx> <ny>`:
  grid size used when regenerating the Experiment 1.3 data. The default is `30 30`.
- `--metrics_13 iterations total_time`:
  metrics used for the performance profiles. The default is both `iterations` and `total_time`.
- `--tau_max_13 <float>`:
  upper limit of the performance-profile ratio axis. The default is `100`.
- `--regenerate_13`:
  forces regeneration of the underlying Experiment 1.1 data before computing Experiment 1.3.

Default output location:

```text
results/experiment_1/
```

Typical output files include:

- `{FUNCTION}_experiment_1_1_heatmap.pdf`
- `{FUNCTION}_experiment_1_1_data.pkl`
- `{FUNCTION}_experiment_1_2.xlsx`
- `{FUNCTION}_experiment_1_2_data.pkl`
- `experiment_1_3_profile_iterations.pdf`
- `experiment_1_3_profile_total_time.pdf`
- `experiment_1_3_results.pkl`

Note that Experiment 1.2 is designed to reuse the dense-grid results from Experiment 1.1 when those files are available.

#### Experiment 2: High-Dimensional Rosenbrock Stress Test

The script [numerical_experiment/experiment_2_rosenbrock.py](numerical_experiment/experiment_2_rosenbrock.py) runs the Rosenbrock stress tests.

Typical usage:

```bash
python3 numerical_experiment/experiment_2_rosenbrock.py --dim 20
python3 numerical_experiment/experiment_2_rosenbrock.py --dim 50 --n_trials 20
python3 numerical_experiment/experiment_2_rosenbrock.py --dim 100 --sdp_tol 1e-4
```

Main options:

- `--dim {5,20,50,100}`:
  dimension of the Rosenbrock problem. The default is `20`.
- `--n_trials <int>`:
  number of random starting points. The default is `10`.
- `--max_iterations <int>`:
  maximum number of iterations. The default is `1000`.
- `--tol <float>`:
  stopping tolerance. The default is `1e-6`.
- `--sdp_tol <float>`:
  fixed SDP tolerance for ALMTON-type methods. The default is `1e-3`.

Default output location:

```text
results/experiment_2/
```

Typical output files include:

- `Rosenbrock-{DIM}_experiment_2.xlsx`
- `Rosenbrock-{DIM}_experiment_2_data.pkl`

#### Experiment 3: Trajectory and Geometric-Structure Tests

The script [numerical_experiment/experiment_3_trajectory_comparison.py](numerical_experiment/experiment_3_trajectory_comparison.py) covers Experiments 3.1, 3.2.1, and 3.2.2.

Typical usage:

```bash
python3 numerical_experiment/experiment_3_trajectory_comparison.py --experiment 3.1 --function Slalom
python3 numerical_experiment/experiment_3_trajectory_comparison.py --experiment 3.1 --function HairpinTurn --x0 0.5 0.0
python3 numerical_experiment/experiment_3_trajectory_comparison.py --experiment 3.2 --function Slalom
python3 numerical_experiment/experiment_3_trajectory_comparison.py --experiment both --function HairpinTurn
```

Main options:

- `--experiment {3.1,3.2,both}`:
  `3.1` runs the trajectory comparison, `3.2` runs the dense-grid and tabular analysis, and `both` runs all Experiment 3 components for the selected function.
- `--function {Slalom,HairpinTurn}`:
  target test function. The default is `Slalom`.
- `--x0 <x> <y>`:
  custom starting point for Experiment 3.1. If omitted, the script uses the default point defined in the code.
- `--max_iterations <int>`:
  maximum number of iterations. The default is `1000`.
- `--tol <float>`:
  stopping tolerance. The default is `1e-8`.
- `--sdp_tol <float>`:
  SDP tolerance for ALMTON-type methods. The default is `1e-8`.
- `--grid_size_3_2 <nx> <ny>`:
  grid size for Experiment 3.2. The default is `20 20`.
- `--n_trials_3_2 <int>`:
  number of fallback random trials for Experiment 3.2.2. The default is `10`.

Default output location:

```text
results/experiment_3/
```

Typical output files include:

- `{FUNCTION}_trajectory_comparison.pdf`
- `{FUNCTION}_trajectory_data.pkl`
- `{FUNCTION}_experiment_3_2_1_heatmap.pdf`
- `{FUNCTION}_experiment_3_2_1_data.pkl`
- `{FUNCTION}_experiment_3_2_2.xlsx`
- `{FUNCTION}_experiment_3_2_2_data.pkl`

#### Modifying the Default Algorithm Parameters

The command-line interface exposes the runtime settings above, but the internal algorithm parameter lists are currently defined directly in the experiment scripts. If you want to change the default parameters for ALMTON, the heuristic ALMTON variant, ALMTON-Interp, AR2-Interp, or AR3-Interp, please edit the corresponding `param_dict` definitions in:

- [numerical_experiment/experiment_1.py](numerical_experiment/experiment_1.py)
- [numerical_experiment/experiment_2_rosenbrock.py](numerical_experiment/experiment_2_rosenbrock.py)
- [numerical_experiment/experiment_3_trajectory_comparison.py](numerical_experiment/experiment_3_trajectory_comparison.py)

In the current codebase, the heuristic ALMTON variant is internally named `almton_heuristic`, even if figures or tables may use a more descriptive display name.

### Citation

If you find this code useful, please cite our paper:

```bibtex
@misc{cai2026globallyconvergentthirdordernewton,
      title={A Globally Convergent Third-Order Newton Method via Unified Semidefinite Programming Subproblems}, 
      author={Yubo Cai and Wenqi Zhu and Coralia Cartis and Gioele Zardini},
      year={2026},
      eprint={2603.09682},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2603.09682}, 
}
```
