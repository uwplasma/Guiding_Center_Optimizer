import jax
import jax.numpy as jnp
from jax import random, jit

def block_diag_jax(*arrs):
    """Construct a block diagonal matrix from given 2D arrays using jax.lax.fori_loop for improved performance."""
    if not arrs:
        return jnp.array([[]])
    shapes = [(a.shape[0], a.shape[1]) for a in arrs]
    total_rows = sum(s[0] for s in shapes)
    total_cols = sum(s[1] for s in shapes)

    def body_fun(i, state):
        out, row_offset, col_offset = state
        a = arrs[i]
        out = jax.lax.dynamic_update_slice(out, a, (row_offset, col_offset))
        return (out, row_offset + a.shape[0], col_offset + a.shape[1])

    init_state = (jnp.zeros((total_rows, total_cols), dtype=arrs[0].dtype), 0, 0)
    out, _, _ = jax.lax.fori_loop(0, len(arrs), body_fun, init_state)
    return out


def pushSeg(nseg, nstep, nus, nc, dt, u0, vstar0, w0, par, s, param, integrator, fJJu):
    """
    Find u, w, vstar, f, J, dJdu on each segment.
    Here we store all values at all time steps for better intuition,
    but this implementation costs a lot of memory.
    See the paper on FD-NILSS for discussion on how to reduce memory cost 
    by computing inner products via only snapshots.
    """

    J = jnp.zeros((nseg, nstep))
    u = jnp.zeros((nseg, nstep, nc))
    f = jnp.zeros(u.shape)
    dJdu = jnp.zeros(u.shape)
    vstar = jnp.zeros(u.shape)
    vstar_perp = jnp.zeros(u.shape)
    w = jnp.zeros((nseg, nstep, nus, nc))
    w_perp = jnp.zeros(w.shape)
    Rs = []  # Rs[0] in code = R_1 in paper
    bs = []  # bs[0] in code = b_1 in paper
   
    # assign initial value, u[0,0], v*[0,0], w[0,0]
    u = u.at[0,0].set(u0)
    vstar = vstar.at[0,0].set(vstar0)
    w = w.at[0,0].set(w0)

    # push forward
    for iseg in range(0, nseg):

        # compute u, w, vstar, f, J, dJdu for current segment
        for istep in range(0, nstep-1):
            u_next, w_next, vstar_next = integrator(u[iseg, istep], w[iseg, istep], vstar[iseg, istep], param, par)
            u = u.at[iseg, istep+1].set(u_next)
            w = w.at[iseg, istep+1].set(w_next)
            vstar = vstar.at[iseg, istep+1].set(vstar_next)
        for istep in range(0, nstep):
            f_val, J_val, dJdu_val = fJJu(u[iseg, istep], par, s)
            f = f.at[iseg, istep].set(f_val)
            J = J.at[iseg, istep].set(J_val)
            dJdu = dJdu.at[iseg, istep].set(dJdu_val)
            
        # calculate vstar_perp and w_perp
        for i in range(0, nstep):
            # Projection: vstar_perp = vstar - (vstar dot f / f dot f)*f
            coef = jnp.dot(vstar[iseg, i], f[iseg, i]) / jnp.dot(f[iseg, i], f[iseg, i])
            vstar_perp = vstar_perp.at[iseg, i].set(vstar[iseg, i] - coef * f[iseg, i])
            for ius in range(0, nus):
                coef_w = jnp.dot(w[iseg, i, ius], f[iseg, i]) / jnp.dot(f[iseg, i], f[iseg, i])
                w_perp = w_perp.at[iseg, i, ius].set(w[iseg, i, ius] - coef_w * f[iseg, i])

        # renormalize at interfaces
        Q_temp, R_temp = jnp.linalg.qr(w_perp[iseg,-1].T, mode='reduced')
        Rs.append(R_temp)
        b_temp = jnp.dot(Q_temp.T, vstar_perp[iseg,-1])
        bs.append(b_temp)
        p_temp = vstar_perp[iseg,-1] - jnp.dot(Q_temp, b_temp)
        if iseg < nseg - 1:
            u = u.at[iseg+1, 0].set(u[iseg, -1])
            w = w.at[iseg+1, 0].set(Q_temp.T)
            vstar = vstar.at[iseg+1, 0].set(p_temp)

    return [u, w, vstar, w_perp, vstar_perp, f, J, dJdu, Rs[:-1], bs[:-1], Q_temp, p_temp]


def nilss(dt, nseg, T_seg, nseg_ps, u0, nus, par, s, param, integrator, fJJu):
    nc = len(u0)
    nstep = int(jnp.round(T_seg / dt)) + 1  # number of step + 1 in each time segment
    
    # push forward u to a stable attractor
    vstar0 = jnp.array([0.0, 0.0, 0.0])
    w0 = jnp.zeros((nus, nc))
    for ius in range(0, nus):
        key = random.PRNGKey(1000 + ius)  # generate a different key for each iteration
        rand_vals = random.uniform(key, shape=(nc,))
        norm = jnp.linalg.norm(rand_vals)
        w0 = w0.at[ius].set(rand_vals / norm)
    u_ps, w_ps, vstar_ps, _, _, _, _, _, _, _, Q_ps, p_ps = pushSeg(nseg_ps, nstep, nus, nc, dt, u0, vstar0, w0, par, s, param, integrator, fJJu)
    u0 = u_ps[-1, -1]
    w0 = Q_ps.T
    vstar0 = p_ps

    # find u, w, vstar on all segments
    u, w, vstar, w_perp, vstar_perp, f, J, dJdu, Rs, bs, _, _ = pushSeg(nseg, nstep, nus, nc, dt, u0, vstar0, w0, par, s, param, integrator, fJJu)

    # a weight matrix for integration, 0.5 at interfaces
    weight = jnp.ones(nstep)
    weight = weight.at[0].set(0.5)
    weight = weight.at[-1].set(0.5)

    # compute Javg
    Javg = jnp.sum(J * weight[jnp.newaxis, :]) / (nstep - 1) / nseg

    # Construct Schur complement of the Lagrange multiplier method of the NILSS problem.
    # See the paper on FD-NILSS for this neat method
    # find C^-1
    Cinvs = []
    for iseg in range(nseg):
        C_iseg = jnp.zeros((nus, nus))
        for i in range(nus):
            for j in range(nus):
                # Note: weight[:, jnp.newaxis] broadcasts correctly.
                C_iseg = C_iseg.at[i, j].set(jnp.sum(w_perp[iseg, :, i, :] * w_perp[iseg, :, j, :] * weight[:, jnp.newaxis]))
        Cinvs.append(jnp.linalg.inv(C_iseg))
    Cinv = block_diag_jax(*Cinvs)

    # construct d
    ds = []
    for iseg in range(nseg):
        d_iseg = jnp.zeros(nus)
        for i in range(nus):
            d_iseg = d_iseg.at[i].set(jnp.sum(w_perp[iseg, :, i, :] * vstar_perp[iseg] * weight[:, jnp.newaxis]))
        ds.append(d_iseg)
    d = jnp.ravel(jnp.array(ds))

    # construct B, first off-diagonal I, then add Rs
    B = jnp.eye((nseg - 1) * nus, nseg * nus, k=nus)
    B = B.at[:, : -nus].add(-block_diag_jax(*Rs))

    # construct b
    b = jnp.ravel(jnp.array(bs))

    # solve
    lbd = jnp.linalg.solve(-B @ Cinv @ B.T, B @ Cinv @ d + b)
    a = -Cinv @ (B.T @ lbd + d)
    a = a.reshape((nseg, nus))

    # calculate v and vperp
    v = jnp.zeros((nseg, nstep, nc))
    v_perp = jnp.zeros(v.shape)
    for iseg in range(nseg):
        v = v.at[iseg].set(vstar[iseg])
        v_perp = v_perp.at[iseg].set(vstar_perp[iseg])
        for ius in range(0, nus):
            v = v.at[iseg].set(v[iseg] + a[iseg, ius] * w[iseg, :, ius, :])
            v_perp = v_perp.at[iseg].set(v_perp[iseg] + a[iseg, ius] * w_perp[iseg, :, ius, :])
    
    # calculate ksi, only need to use last step in each segment
    ksi = jnp.zeros((nseg, nstep))
    for iseg in range(nseg):
        for i in (0, -1):
            num = jnp.dot(v[iseg, i], f[iseg, i])
            den = jnp.dot(f[iseg, i], f[iseg, i])
            ksi = ksi.at[iseg, i].set(num / den)
        # print(abs(ksi[iseg, 0]))
        assert jnp.abs(ksi[iseg, 0]) <= 1e-5

    # compute dJds
    dJdss = []
    for iseg in range(nseg):
        t1 = jnp.sum(dJdu[iseg] * v[iseg] * weight[:, jnp.newaxis]) / (nstep - 1) / nseg
        t2 = ksi[iseg, -1] * (Javg - J[iseg, -1]) / (nstep - 1) / nseg / dt
        dJdss.append(t1 + t2)
    dJds = jnp.sum(jnp.array(dJdss))

    return Javg, dJds