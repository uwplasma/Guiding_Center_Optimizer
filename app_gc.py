from functools import partial
import jax 
import jax.numpy as jnp
from jax import jit, jacobian
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilss import *
import argparse
import os


a0_global = 0.1
a1_global = 1.0
iota_global = 0.5
lam_global = 0.1
G_global = 1.0
default_params = {'a0': 0.1, 'a1': 1.0, 'iota': 0.5, 'G': 1.0, 'lam': 0.1}
dt = 0.001

# precomputed_derivatives = None
# precomputed_spatial_derivatives = None

# def set_params(par, value):
#     global a0_global, a1_global, iota_global, lam_global, G_global
#     if par == "a0":
#         a0_global = value
#     elif par == "a1":
#         a1_global = value
#     elif par == "iota":
#         iota_global = value
#     elif par == "lambda":
#         lam_global = value
#     elif par == "G":
#         G_global = value

# def compute_derivatives():
#     x, y, z, a0, a1, iota, G, lambda_ = sp.symbols('x y z a0 a1 iota G lambda', real=True)
#     B = 1 + a0 * sp.sqrt(x) * sp.cos(y - a1*z)
#     V = sp.sqrt(1 - lambda_ * B)

#     df_dx = - (1/B) * sp.diff(B, y) * (2/lambda_ - B)
#     df_dy = (1/B) * sp.diff(B, x) * (2/lambda_ - B) + (iota * V * B) / G
#     df_dz = (V * B) / G

#     parameters = (x, y, z, a0, a1, iota, G, lambda_)
#     derivatives = {
#         'a0': [sp.lambdify(parameters, sp.diff(df_dx, a0), "numpy"),
#                sp.lambdify(parameters, sp.diff(df_dy, a0), "numpy"),
#                sp.lambdify(parameters, sp.diff(df_dz, a0), "numpy")],
        
#         'a1': [sp.lambdify(parameters, sp.diff(df_dx, a1), "numpy"),
#                sp.lambdify(parameters, sp.diff(df_dy, a1), "numpy"),
#                sp.lambdify(parameters, sp.diff(df_dz, a1), "numpy")],
        
#         'iota': [sp.lambdify(parameters, sp.diff(df_dx, iota), "numpy"),
#                  sp.lambdify(parameters, sp.diff(df_dy, iota), "numpy"),
#                  sp.lambdify(parameters, sp.diff(df_dz, iota), "numpy")],
        
#         'G': [sp.lambdify(parameters, sp.diff(df_dx, G), "numpy"),
#               sp.lambdify(parameters, sp.diff(df_dy, G), "numpy"),
#               sp.lambdify(parameters, sp.diff(df_dz, G), "numpy")],
        
#         'lambda': [sp.lambdify(parameters, sp.diff(df_dx, lambda_), "numpy"),
#                    sp.lambdify(parameters, sp.diff(df_dy, lambda_), "numpy"),
#                    sp.lambdify(parameters, sp.diff(df_dz, lambda_), "numpy")]
#     }
#     return derivatives

# def compute_spatial_derivatives():
#     x, y, z, a0, a1, iota, G, lambda_ = sp.symbols('x y z a0 a1 iota G lambda', real=True)
#     B = 1 + a0 * sp.sqrt(x) * sp.cos(y - a1*z)
#     V = sp.sqrt(1 - lambda_ * B)

#     df_dx = - (1/B) * sp.diff(B, y) * (2/lambda_ - B)
#     df_dy = (1/B) * sp.diff(B, x) * (2/lambda_ - B) + (iota * V * B) / G
#     df_dz = (V * B) / G

#     parameters = (x, y, z, a0, a1, iota, G, lambda_)
#     spatial_derivatives = np.array([
#         [sp.lambdify(parameters, sp.diff(df_dx, x), "numpy"),
#          sp.lambdify(parameters, sp.diff(df_dy, x), "numpy"),
#          sp.lambdify(parameters, sp.diff(df_dz, x), "numpy")],

#         [sp.lambdify(parameters, sp.diff(df_dx, y), "numpy"),
#          sp.lambdify(parameters, sp.diff(df_dy, y), "numpy"),
#          sp.lambdify(parameters, sp.diff(df_dz, y), "numpy")],

#         [sp.lambdify(parameters, sp.diff(df_dx, z), "numpy"),
#          sp.lambdify(parameters, sp.diff(df_dy, z), "numpy"),
#          sp.lambdify(parameters, sp.diff(df_dz, z), "numpy")]
#     ])
#     return spatial_derivatives


# precomputed_derivatives = compute_derivatives()
# precomputed_spatial_derivatives = compute_spatial_derivatives()

# def B_func(x, y, z):
#     return 1.0 + a0_global * np.sqrt(x) * np.cos(y - a1_global*z)

# def V_func(x, y, z):
#     return np.sqrt(1.0 - lam_global * B_func(x, y, z))
def B_func(x, y, z, a0, a1, lam):
    # Compute the B field
    return 1.0 + a0 * jnp.sqrt(x) * jnp.cos(y - a1 * z)

def V_func(x, y, z, G, lam, B):
    # Compute V based on B
    return jnp.sqrt(1.0 - lam * B)


# def f_ode(x, y, z):
#     B = B_func(x, y, z)
#     invB = 1.0 / B
#     factor = 2.0 / lam_global - B
    
#     dBdy = -a0_global * np.sqrt(x) * np.sin(y - a1_global*z)
#     dfdx = -invB * dBdy * factor
    
#     dBdx = a0_global * (1.0 / (2.0 * np.sqrt(x))) * np.cos(y - a1_global*z)
#     dfdy = invB * dBdx * factor + (iota_global * V_func(x, y, z) * B) / G_global
    
#     dfdz = (B * V_func(x, y, z)) / G_global
    
#     return np.array([dfdx, dfdy, dfdz])
def f_ode(x, y, z, a0, a1, iota, G, lam):
    B = B_func(x, y, z, a0, a1, lam)
    invB = 1.0 / B
    factor = 2.0 / lam - B
    dBdy = -a0 * jnp.sqrt(x) * jnp.sin(y - a1 * z)
    dfdx = -invB * dBdy * factor
    dBdx = a0 * (1.0 / (2.0 * jnp.sqrt(x))) * jnp.cos(y - a1 * z)
    V = V_func(x, y, z, G, lam, B)
    dfdy = invB * dBdx * factor + (iota * V * B) / G
    dfdz = (B * V) / G
    return jnp.array([dfdx, dfdy, dfdz])

def f_ode_wrapper(u, params):
    x, y, z = u
    return f_ode(x, y, z, params['a0'], params['a1'], params['iota'], params['G'], params['lam'])

# def ddt(uwvs, par, value):
#     set_params(par, value)
#     u = np.asarray(uwvs[0])
#     w = uwvs[1]
#     vstar = uwvs[2]
#     x, y, z = u
    
#     param_values = (x, y, z, a0_global, a1_global, iota_global, G_global, lam_global)

#     dfdpar = np.array([func(*param_values) for func in precomputed_derivatives[par]])
#     Df = np.array([[func(*param_values) for func in row] for row in precomputed_spatial_derivatives])

#     dudt = f_ode(x, y, z)
#     dwdt = np.dot(Df, w.T)
#     dvstardt = np.dot(Df, vstar) + dfdpar
    
#     return [dudt, dwdt.T, dvstardt]

def ddt(uwvs, params, par):
    u, w, vstar = uwvs
    dudt = f_ode_wrapper(u, params)
    Df = jacobian(f_ode_wrapper, argnums=0)(u, params)
    dwdt = jnp.dot(Df, w.T).T
    def f_par(p_val):
        new_params = params.copy()
        new_params[par] = p_val
        return f_ode_wrapper(u, new_params)
    dfdpar = jacobian(f_par)(params[par])
    dvstardt = jnp.dot(Df, vstar) + dfdpar
    return (dudt, dwdt, dvstardt)


# def fJJu(u, par, value):
#     set_params(par, value)
#     x, y, z = u
#     f_val = f_ode(x, y, z)
#     return f_val, x, np.array([1, 0, 0])
def fJJu(u, params):
    f_val = f_ode_wrapper(u, params)
    return f_val, u[0], jnp.array([1, 0, 0])

def fJJu_wrapper(u, par, value):
    new_params = default_params.copy()
    new_params[par] = value
    return fJJu(u, new_params)

# def RK4(u, w, vstar, par, value):
#     uwvs = [u, w, vstar]

#     k0 = [dt * vec for vec in ddt(uwvs, par, value)]
#     k1 = [dt * vec for vec in ddt([uwvs[i] + 0.5*k0[i] for i in range(3)], par, value)]
#     k2 = [dt * vec for vec in ddt([uwvs[i] + 0.5*k1[i] for i in range(3)], par, value)]
#     k3 = [dt * vec for vec in ddt([uwvs[i] + k2[i] for i in range(3)], par, value)]
#     uwvs_new = [uwvs[i] + (k0[i] + 2*k1[i] + 2*k2[i] + k3[i]) / 6.0 for i in range(3)]
#     return uwvs_new

@partial(jax.jit, static_argnames=['par'])
def RK4(u, w, vstar, params, par):
    uwvs = [u, w, vstar]
    k0 = tuple(dt * comp for comp in ddt(uwvs, params, par))
    uwvs_half = tuple(uwvs[i] + 0.5 * k0[i] for i in range(3))
    k1 = tuple(dt * comp for comp in ddt(uwvs_half, params, par))
    uwvs_half2 = tuple(uwvs[i] + 0.5 * k1[i] for i in range(3))
    k2 = tuple(dt * comp for comp in ddt(uwvs_half2, params, par))
    uwvs_full = tuple(uwvs[i] + k2[i] for i in range(3))
    k3 = tuple(dt * comp for comp in ddt(uwvs_full, params, par))
    new_uwvs = tuple(uwvs[i] + (k0[i] + 2 * k1[i] + 2 * k2[i] + k3[i]) / 6.0 for i in range(3))
    return new_uwvs


def Euler(u, w, vstar, par, value):
    uwvs = [u, w, vstar]
    k0 = [dt*vec for vec in ddt(uwvs, par, value)] 
    uwvs_new = [v1+v2 for v1,v2 in zip(uwvs,k0)] 
    return uwvs_new

def main():
    parser = argparse.ArgumentParser(description='Run NILSS sensitivity analysis.')
    parser.add_argument('--par', type=str, required=True,
                        choices=['a0', 'a1', 'iota', 'lam', 'G'],
                        help='Parameter to vary')

    args = parser.parse_args()

    nseg = 300
    T_seg = 0.01
    nseg_ps = 100
    nc = 3
    nus = 1

    # global precomputed_derivatives, precomputed_spatial_derivatives
    # precomputed_derivatives = compute_derivatives()
    # precomputed_spatial_derivatives = compute_spatial_derivatives()

    par = args.par
    par_limit = {
        "a0": (0.05, 0.25),
        "a1": (0.9, 1.1),
        "iota": (0.4, 0.6),
        "lam": (0.05, 0.25),
        "G": (0.9, 1.1)
    }

    par_lb, par_ub = par_limit[par]
    step_size = 0.01
    num_steps = int(jnp.round((par_ub - par_lb) / step_size)) + 1
    par_arr = jnp.linspace(par_lb, par_ub, num_steps)

    # J_arr = np.zeros(par_arr.shape)
    # dJdpar_arr = np.zeros(par_arr.shape)

    # # TODO: optimize with JAX vmap
    # for i, par_value in enumerate(par_arr):
    #     x = 0.1 * np.random.rand()
    #     y = 2.0 * np.pi * np.random.rand()
    #     z = 2.0 * np.pi * np.random.rand()
    #     u0 = np.array([x, y, z])
    #     print(f'{par}={par_value}, u0={u0}')

    #     J_val, dJdpar_val = nilss(dt, nseg, T_seg, nseg_ps, u0, nus, par, par_value, RK4, fJJu)
    #     J_arr[i] = J_val
    #     dJdpar_arr[i] = dJdpar_val


    # TODO: make nilss() jax-ify to use vmap in the later procedure
    # TODO: remove par and par_value later

    def compute_sensitivity(par_value, key):
        key, subkey = jax.random.split(key)
        x = 0.1 * jax.random.uniform(subkey)
        key, subkey = jax.random.split(key)
        y = 2.0 * jax.random.uniform(subkey) * 2 * jnp.pi
        key, subkey = jax.random.split(key)
        z = 2.0 * jax.random.uniform(subkey) * 2 * jnp.pi
        u0 = jnp.array([x, y, z])
        print(f'{par} = {par_value}, u0 = {u0}')
        params = default_params.copy()
        params[par] = par_value
        J_val, dJdpar_val = nilss(dt, nseg, T_seg, nseg_ps, u0, nus, par, float(par_value), params, RK4, fJJu_wrapper)
        return J_val, dJdpar_val, key

    key = jax.random.PRNGKey(20250407)
    J_arr = []
    dJdpar_arr = []
    for par_value in par_arr:
        J_val, dJdpar_val, key = compute_sensitivity(par_value, key)
        J_arr.append(J_val)
        dJdpar_arr.append(dJdpar_val)
    J_arr = jnp.array(J_arr)
    dJdpar_arr = jnp.array(dJdpar_arr)


    output_dir = "nilss_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    par_latex = {
        'a0': r'a_0',
        'a1': r'a_1',
        'lam': r'\lambda',
        'iota': r'\iota',
        'G': r'G'
    }
    par_label = par_latex.get(par, par)

    plt.figure(figsize=(12, 12))

    # Plot for time-averaged x
    plt.subplot(2, 1, 1)
    plt.plot(par_arr, J_arr)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)  # horizontal line at y=0
    plt.xlabel(fr'${par_label}$')
    plt.ylabel(r'$\langle J \rangle$')

    # Plot for sensitivity of J
    plt.subplot(2, 1, 2)
    plt.plot(par_arr, dJdpar_arr, marker='s', linestyle='-')
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)  # horizontal line at y=0
    plt.xlabel(fr'${par_label}$')
    plt.ylabel(fr"$d \langle J \rangle /d {par_label}$", fontsize=14)

    save_path = os.path.join(output_dir, f'guiding_center_{par}.png')
    plt.savefig(save_path)

if __name__ == '__main__':
    main()