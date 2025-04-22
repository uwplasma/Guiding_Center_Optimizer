import os
import numpy as np
from scipy.optimize import minimize
from nilss import nilss
from app_gc import RK4, Euler, fJJu
import argparse

np.random.seed(20250402)

def optimize_guiding_center_nilss(par_name, par_bounds, u0, nus, dt, nseg, T_seg, nseg_ps, integrator, fJJu, maxiter = 100, tol = 1e-6):
    def objective(par_value):
        J, dJdpar = nilss(
            dt=dt,
            nseg=nseg,
            T_seg=T_seg,
            nseg_ps=nseg_ps,
            u0=u0,
            nus=nus,
            par=par_name,
            s=par_value,
            integrator=integrator,
            fJJu=fJJu
        )
        return J, dJdpar

    def scipy_objective(par_value):
        J, dJdpar = objective(par_value[0])
        return J, np.array([dJdpar])
    
    result = minimize(
        fun=scipy_objective,
        x0=[(par_bounds[0] + par_bounds[1]) / 2],
        jac=True,
        method='L-BFGS-B',
        bounds=[par_bounds],
        options={
            # 'maxiter': 20,
            # 'ftol': 1e-2,
            'disp': True
        }
    )

    return result


def main():
    parser = argparse.ArgumentParser(description='Run NILSS sensitivity analysis.')
    parser.add_argument('--par', type=str, required=True,
                        choices=['a0', 'a1', 'iota', 'lambda', 'G'],
                        help='Parameter to vary')

    args = parser.parse_args()
    par = args.par
    par_limit = {
        "a0": (0.05, 0.25),
        "a1": (0.9, 1.1),
        "iota": (0.4, 0.6),
        "lambda": (0.01, 0.21),
        "G": (0.9, 1.1)
    }
    nseg = 200
    T_seg = 0.01
    nseg_ps = 200
    nus = 1
    dt = 0.0001
    x = 0.1 * np.random.rand()
    y = 2.0 * np.pi * np.random.rand()
    z = 2.0 * np.pi * np.random.rand()
    u0 = np.array([x, y, z])
    
    result = optimize_guiding_center_nilss(par, par_limit[par], u0, nus, dt, nseg, T_seg, nseg_ps, RK4, fJJu)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    text_result_path = os.path.join(results_dir, f"optimization_guiding_center_results_{par}.txt")
    with open(text_result_path, "w") as f:
        f.write("Optimization Result:\n")
        f.write(f"  Optimal {par}: {result.x[0]:.4f}\n")
        f.write(f"  Minimum cost J: {result.fun:.4e}\n")


if __name__ == '__main__':
    main()