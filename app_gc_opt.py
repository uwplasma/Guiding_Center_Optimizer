from functools import partial
import os
import argparse
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
from nilss import nilss
from app_gc import RK4, Euler, fJJu_wrapper, default_params

def objective_fn(par_value, given_params, u0, nus, dt, nseg, T_seg, nseg_ps, integrator, fJJu, par_name):
    par_scalar = par_value[0]
    params = given_params.copy()
    params[par_name] = par_scalar
    J, dJdpar = nilss(
        dt=dt,
        nseg=nseg,
        T_seg=T_seg,
        nseg_ps=nseg_ps,
        u0=u0,
        nus=nus,
        par=par_name,
        param=params,
        s=par_scalar,
        integrator=integrator,
        fJJu=fJJu
    )
    print("##############")
    print(" Running NILSS with parameter:", par_name, "=", par_scalar)
    print(" Cost function value (J):", J)
    print(" Gradient (dJ/dpar):", dJdpar)
    return J, dJdpar

def jax_optimize_guiding_center_nilss(par_name, par_bounds, given_params, u0, nus, dt, nseg, T_seg, nseg_ps, integrator, fJJu, maxiter=5):
    def wrapped_obj(par_value):
        J, dJdpar = objective_fn(par_value, given_params, u0, nus, dt, nseg, T_seg, nseg_ps, integrator, fJJu, par_name)
        return -J, -dJdpar

    # Initialize the parameter
    key = jax.random.PRNGKey(20250503)
    init_par = jax.random.uniform(key, shape=(1,), minval=par_bounds[0], maxval=par_bounds[1])

    # Define the optimizer
    optimizer = optax.scale_by_lbfgs()
    opt_state = optimizer.init(init_par)

    par = init_par
    for i in range(maxiter):
        J, grad = wrapped_obj(par)
        grad = jnp.reshape(grad, par.shape)
        updates, opt_state = optimizer.update(grad, opt_state, par)
        par = optax.apply_updates(par, updates)
        par = jnp.clip(par, par_bounds[0], par_bounds[1])

        print(f"Iteration {i + 1}, J: {J}, par: {par}")

        # Convergence check
        if jnp.linalg.norm(grad) < 1e-6:
            print("Converged!")
            break

    return par, J

def main():
    parser = argparse.ArgumentParser(description='Run optimization based on NILSS.')
    parser.add_argument('--par', type=str, required=True,
                        choices=['a0', 'a1', 'iota', 'lambda', 'G'],
                        help='Parameter to vary')
    args = parser.parse_args()
    par = args.par

    par_limit = {
        "a0": (0.05, 0.25),
        "a1": (0.9, 1.1),
        "iota": (0.4, 0.6),
        "lambda": (0.05, 0.25),
        "G": (0.9, 1.1)
    }

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
    
    result_par, result_J = jax_optimize_guiding_center_nilss(par, par_limit[par], default_params, u0, nus, dt, nseg, T_seg, nseg_ps, RK4, fJJu_wrapper)
    
    output_dir = "optimization_results"
    os.makedirs(output_dir, exist_ok=True)
    text_output_path = os.path.join(output_dir, f"optimization_guiding_center_results_{par}.txt")
    with open(text_output_path, "w") as f:
        f.write("Optimization Result:\n")
        f.write(f"  Optimal {par}: {result_par[0]:.4f}\n")
        f.write(f"  Minimum J: {result_J:.4e}\n")

if __name__ == '__main__':
    main()