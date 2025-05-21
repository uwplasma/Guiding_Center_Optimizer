from functools import partial
import os
import argparse
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
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
            print("Converged (change in J < 1%)!")
            break
        prev_J = J

    return par, J

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
    
    result_par, result_J = jax_optimize_guiding_center_nilss(
        par_names, par_bounds, default_params, u0, nus, dt, nseg, T_seg, nseg_ps, RK4, fJJu_wrapper
    )
    
    output_dir = "optimization_results"
    os.makedirs(output_dir, exist_ok=True)
    text_output_path = os.path.join(output_dir, f"optimization_guiding_center_results_{'_'.join(par_names)}.txt")
    with open(text_output_path, "w") as f:
        f.write("Optimization Result:\n")
        for i, par_name in enumerate(par_names):
            f.write(f"  Optimal {par_name}: {result_par[i]:.4f}\n")
        f.write(f"  Minimum J: {result_J:.4e}\n")

if __name__ == '__main__':
    main()