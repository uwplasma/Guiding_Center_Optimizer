from functools import partial
import os
import argparse
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from nilss import nilss
from app_gc import RK4, Euler, fJJu_wrapper, default_params

def objective_fn(par_values, given_params, u0, nus, dt, nseg, T_seg, nseg_ps, integrator, fJJu, par_names):
    params = given_params.copy()
    for i, par_name in enumerate(par_names):
        params[par_name] = par_values[i]
    if len(par_names) == 1:
        J, dJdpar = nilss(
            dt=dt,
            nseg=nseg,
            T_seg=T_seg,
            nseg_ps=nseg_ps,
            u0=u0,
            nus=nus,
            par=par_names[0],
            param=params,
            s=par_values,
            integrator=integrator,
            fJJu=fJJu
        )
        grad = jnp.array([dJdpar])
    else:
        J_list = []
        grad_list = []
        for i, par_name in enumerate(par_names):
            J, dJdpar = nilss(
                dt=dt,
                nseg=nseg,
                T_seg=T_seg,
                nseg_ps=nseg_ps,
                u0=u0,
                nus=nus,
                par=par_name,
                param=params,
                s=par_values[i],
                integrator=integrator,
                fJJu=fJJu
            )
            J_list.append(J)
            grad_list.append(dJdpar)
        J = jnp.mean(jnp.array(J_list))
        grad = jnp.array(grad_list)

    print("##############")
    print(" Running NILSS with parameters:", {par_names[i]: par_values[i] for i in range(len(par_names))})
    print(" Cost function value (J):", J)
    print(" Gradient (dJ/dpar):", grad)
    return J, grad

def finite_difference_gradient(par_values, given_params, u0, nus, dt, nseg, T_seg, nseg_ps, integrator, fJJu, par_names, h=1e-4):
    """Compute cost and gradient using 2-point finite difference."""
    params = given_params.copy()
    for i, par_name in enumerate(par_names):
        params[par_name] = par_values[i]
    J0, _ = nilss(
        dt=dt,
        nseg=nseg,
        T_seg=T_seg,
        nseg_ps=nseg_ps,
        u0=u0,
        nus=nus,
        par=par_names[0],
        param=params,
        s=par_values[0],
        integrator=integrator,
        fJJu=fJJu
    )
    grad = []
    for i, par_name in enumerate(par_names):
        par_values_pert = par_values.copy()
        par_values_pert = par_values_pert.at[i].set(par_values[i] + h)
        params_pert = given_params.copy()
        for j, pn in enumerate(par_names):
            params_pert[pn] = par_values_pert[j]
        J1, _ = nilss(
            dt=dt,
            nseg=nseg,
            T_seg=T_seg,
            nseg_ps=nseg_ps,
            u0=u0,
            nus=nus,
            par=par_names[0],
            param=params_pert,
            s=par_values_pert[0],
            integrator=integrator,
            fJJu=fJJu
        )
        grad.append((J1 - J0) / h)
    return J0, jnp.array(grad)

def jax_optimize_guiding_center_nilss(par_names, par_bounds, given_params, u0, nus, dt, nseg, T_seg, nseg_ps, integrator, fJJu, maxiter = 20):
    def wrapped_obj(par_values):
        J, dJdpar = objective_fn(par_values, given_params, u0, nus, dt, nseg, T_seg, nseg_ps, integrator, fJJu, par_names)
        return -J, -dJdpar

    key = jax.random.PRNGKey(20250503)
    minval = jnp.array([b[0] for b in par_bounds])
    maxval = jnp.array([b[1] for b in par_bounds])
    init_par = jax.random.uniform(key, shape=(len(par_names),), minval=minval, maxval=maxval)

    optimizer = optax.scale_by_lbfgs()
    opt_state = optimizer.init(init_par)

    par = init_par
    prev_J = None
    for i in range(maxiter):
        J, grad = wrapped_obj(par)
        updates, opt_state = optimizer.update(grad, opt_state, par)
        par = optax.apply_updates(par, updates)
        par = jnp.clip(par, minval, maxval)

        print(f"Iteration {i + 1}, J: {J}, par: {par}")

        if prev_J is not None and jnp.abs((J - prev_J) / prev_J) < 0.01:
            print("[nilss] Converged (change in J < 1%)!")
            break
        prev_J = J

    return par, J

def jax_optimize_guiding_center_2point(par_names, par_bounds, given_params, u0, nus, dt, nseg, T_seg, nseg_ps, integrator, fJJu, maxiter = 20):
    def wrapped_obj(par_values):
        J, grad = finite_difference_gradient(par_values, given_params, u0, nus, dt, nseg, T_seg, nseg_ps, integrator, fJJu, par_names)
        return -J, -grad

    key = jax.random.PRNGKey(20250503)
    minval = jnp.array([b[0] for b in par_bounds])
    maxval = jnp.array([b[1] for b in par_bounds])
    init_par = jax.random.uniform(key, shape=(len(par_names),), minval=minval, maxval=maxval)

    optimizer = optax.scale_by_lbfgs()
    opt_state = optimizer.init(init_par)

    par = init_par
    prev_J = None
    for i in range(maxiter):
        J, grad = wrapped_obj(par)
        updates, opt_state = optimizer.update(grad, opt_state, par)
        par = optax.apply_updates(par, updates)
        par = jnp.clip(par, minval, maxval)

        print(f"[2-point] Iteration {i + 1}, J: {J}, par: {par}")

        if prev_J is not None and jnp.abs((J - prev_J) / prev_J) < 0.01:
            print("[2-point] Converged (change in J < 1%)!")
            break
        prev_J = J

    return par, J

def plot_comparison(par_names, par_bounds, given_params, u0, nus, dt, nseg, T_seg, nseg_ps, integrator, fJJu, nilss_opt, twopoint_opt, output_dir):
    # 1D only for plotting
    if len(par_names) != 1:
        return
    par_name = par_names[0]
    par_min, par_max = par_bounds[0]
    par_grid = jnp.linspace(par_min, par_max, 20)
    J_grid = []
    for val in par_grid:
        params = given_params.copy()
        params[par_name] = val
        J, _ = nilss(
            dt=dt,
            nseg=nseg,
            T_seg=T_seg,
            nseg_ps=nseg_ps,
            u0=u0,
            nus=nus,
            par=par_name,
            param=params,
            s=val,
            integrator=integrator,
            fJJu=fJJu
        )
        J_grid.append(J)
    J_grid = jnp.array(J_grid)

    plt.figure(figsize=(8,6))
    plt.plot(par_grid, J_grid, label="Cost function")
    plt.axvline(x=float(nilss_opt[0][0]), color='r', linestyle='--', label='NILSS optimum')
    plt.axvline(x=float(twopoint_opt[0][0]), color='b', linestyle='--', label='2-point optimum')
    plt.xlabel(par_name)
    plt.ylabel("Cost function J")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"optimization_comparison_{par_name}.png"))
    plt.close()
    print("figure saved to", os.path.join(output_dir, f"optimization_comparison_{par_name}.png"))

def main():
    parser = argparse.ArgumentParser(description='Run optimization based on NILSS.')
    parser.add_argument('--par', type=str, nargs='+', required=True,
                        choices=['a0', 'a1', 'iota', 'lambda', 'G'],
                        help='Parameters to vary (space-separated list)')
    args = parser.parse_args()
    par_names = args.par

    par_limit = {
        "a0": (0.05, 0.25),
        "a1": (0.9, 1.1),
        "iota": (0.4, 0.6),
        "lambda": (0.05, 0.25),
        "G": (0.9, 1.1)
    }

    par_bounds = [par_limit[par_name] for par_name in par_names]

    nseg = 300
    T_seg = 0.01
    nseg_ps = 100
    nus = 1
    dt = 0.001

    key = jax.random.PRNGKey(20240503)
    x = 0.1 * jax.random.uniform(key)
    y = 2.0 * jnp.pi * jax.random.uniform(key)
    z = 2.0 * jnp.pi * jax.random.uniform(key)
    u0 = jnp.array([x, y, z])

    print(f"Initial condition: {u0}")

    result_par_nilss, result_J_nilss = jax_optimize_guiding_center_nilss(
        par_names, par_bounds, default_params, u0, nus, dt, nseg, T_seg, nseg_ps, RK4, fJJu_wrapper
    )
    result_par_2point, result_J_2point = jax_optimize_guiding_center_2point(
        par_names, par_bounds, default_params, u0, nus, dt, nseg, T_seg, nseg_ps, RK4, fJJu_wrapper
    )

    output_dir = "optimization_results"
    os.makedirs(output_dir, exist_ok=True)
    text_output_path = os.path.join(output_dir, f"optimization_guiding_center_results_{'_'.join(par_names)}.txt")
    with open(text_output_path, "w") as f:
        f.write("Optimization Result (NILSS):\n")
        for i, par_name in enumerate(par_names):
            f.write(f"  NILSS optimal {par_name}: {float(result_par_nilss[i]):.4f}\n")
        f.write(f"  NILSS minimum J: {float(result_J_nilss):.4e}\n\n")
        f.write("Optimization Result (2-point):\n")
        for i, par_name in enumerate(par_names):
            f.write(f"  2-point optimal {par_name}: {float(result_par_2point[i]):.4f}\n")
        f.write(f"  2-point minimum J: {float(result_J_2point):.4e}\n")

    plot_comparison(
        par_names, par_bounds, default_params, u0, nus, dt, nseg, T_seg, nseg_ps, RK4, fJJu_wrapper,
        (result_par_nilss, result_J_nilss), (result_par_2point, result_J_2point), output_dir
    )

if __name__ == '__main__':
    main()