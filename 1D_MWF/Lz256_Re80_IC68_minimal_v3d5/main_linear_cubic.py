"""
main_linear_cubic.py
====================
1D Waleffe-flow model (tilted-band geometry) with LINEAR-CUBIC drag.

Drag law:  F_drag = -alpha * (u + c_lc * |u|^2 * u) / norm_lc
           where  norm_lc = 1 + c_lc  so that h(1) = 1 exactly.

Because h(1) = 1, the body force that sustains the laminar state is
IDENTICAL to the linear-drag case:
   f(y) = (alpha + beta^2/Re) * sin(beta*y)
No change to params.py is needed for the body force.

What DOES change versus linear drag:
  1. The effective drag coefficient on each perturbation mode shifts
     because we expand (U_lam + u_perturb)^3 around the laminar state.
  2. Quadratic and cubic cross-terms appear on the RHS of u1 and zeta.
  3. A pure cubic term appears on the RHS of u0.
  4. The dealiasing factor must be raised from 3/2 to 2 to avoid
     aliasing errors from the cubic products.

Parameters needed in params.py (in addition to the usual ones):
   c_lc  (float) — cubic correction coefficient, e.g. 0.1
"""

import numpy as np
import h5py
from mpi4py import MPI

comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()

import time
import pathlib
from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

ncores = comm.Get_size()
nnodes = int(ncores / 16)
logger.info('Running on %s cores, %s nodes' % (ncores, nnodes))

from params import *

logger.info('Lz = %.7f, Re = %.4e' % (Lz, Re))

import spec1d
import flux1d
import nonlin_term
import vector_cal as vc

# ══════════════════════════════════════════════════════════════════════════════
# DRAG PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# c_lc is the cubic correction coefficient.
# Setting c_lc=0.1 means the cubic term is 10% as strong as the linear term
# at u=1.  Increasing this increases the nonlinear correction strength.
c_lc = 0.1

# norm_lc ensures h(1) = (1 + c_lc) / norm_lc = 1 exactly.
# This is critical: if h(1) != 1 the body force no longer sustains the
# laminar state and the laminar solution will drift away from u=1.
norm_lc = 1.0 + c_lc    # = 1.1

# Log the key parameters so we can verify them in the output files
# logger.info('Drag: linear-cubic, c_lc=%.3f, norm_lc=%.3f' % (c_lc, norm_lc))
# logger.info('alpha * (1.0 + 3.0*c_lc * ulam**2) / norm_lc=%.5f, alpha * (1.0 + 3.0*c_lc * wlam**2) / norm_lc=%.5f, 3.0 * alpha / norm_lc=%.5f'
#             % (alpha * (1.0 + 3.0*c_lc * ulam**2) / norm_lc, alpha * (1.0 + 3.0*c_lc * wlam**2) / norm_lc, 3.0 * alpha / norm_lc))
# logger.info('3.0 * alpha * c_lc * ulam / norm_lc=%.5f, 3.0 * alpha * c_lc * wlam / norm_lc=%.5f' % (3.0 * alpha * c_lc * ulam / norm_lc, 3.0 * alpha * c_lc * wlam / norm_lc))
# logger.info('alpha * c_lc / norm_lc=%.5f, 3.0 * alpha * c_lc / norm_lc=%.5f' % (alpha * c_lc / norm_lc, 3.0 * alpha * c_lc / norm_lc))

# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN
# ══════════════════════════════════════════════════════════════════════════════

# IMPORTANT CHANGE: dealias raised from 3/2 to 2.
#
# With 3/2 dealiasing, the product of two fields with Nz/2 modes is correctly
# represented (Nz/2 + Nz/2 = Nz, and 3/2 * Nz/2 > Nz).
# But a cubic product (u1^3) has energy at wavenumbers up to 3*(Nz/2).
# With 3/2 dealiasing, 3*(Nz/2) > 3/2*(Nz/2), so the high-k energy aliases
# back into the resolved modes and corrupts the solution every timestep.
# With dealias=2, the dealiased grid has 2*Nz points, safely representing
# products up to wavenumber 2*(Nz/2) = Nz. Since 3*(Nz/2)/2 = 3*Nz/4 < Nz
# this is still not perfect for u1^3, but dealias=2 is the standard choice
# for cubic nonlinearities in pseudospectral codes and is a significant
# improvement over 3/2.
z_basis = de.Fourier('z', Nz, interval=(0, Lz), dealias=2)
domain  = de.Domain([z_basis], grid_dtype=np.float64)

# Evaluate grid at the dealiased scale (must match dealias factor)
z  = z_basis.grid(scale=2)
kz = domain.elements(0)
k2 = kz**2
S  = (beta - np.sin(beta)*np.cos(beta)) / (2*beta)

# Laminar flow amplitudes (constants in z — the across-band coordinate)
ulam = np.cos(theta)    # along-band laminar amplitude
wlam = np.sin(theta)    # across-band laminar amplitude

# Timestep
kcut = np.max(kz)
if rand_force:
    dt = dt_init / 4
else:
    dt = dt_init

# ══════════════════════════════════════════════════════════════════════════════
# RANDOM FORCING (optional, unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════
if rand_force:
    def my_forcing(*args):
        q      = args[0].data
        mu     = args[1].value
        sigma  = args[2].value
        deltat = args[3].value
        return np.sqrt(deltat) * q * np.random.normal(mu, sigma, size=q.shape)
    def Forcing(*args, domain=domain, F=my_forcing):
        return de.operators.GeneralFunction(domain, layout='g', func=F, args=args)

# ══════════════════════════════════════════════════════════════════════════════
# PROBLEM SETUP
# ══════════════════════════════════════════════════════════════════════════════
problem = de.IVP(domain, variables=['u0','u1','v1','w1','zeta','q0'])

# ── Standard parameters (unchanged from linear drag) ──────────────────────────
problem.parameters['Lz']    = Lz
problem.parameters['S']     = S
problem.parameters['beta']  = beta
problem.parameters['theta'] = theta
problem.parameters['a']     = a
problem.parameters['eta']   = eta
problem.parameters['Re']    = Re
problem.parameters['c1']    = c1
problem.parameters['C_zt']  = C_zt
problem.parameters['alpha'] = alpha
problem.parameters['kappa'] = kappa
problem.parameters['ulam']  = ulam
problem.parameters['wlam']  = wlam
problem.parameters['c_lc']    = c_lc
problem.parameters['norm_lc'] = norm_lc
if rand_force:
    de.operators.parseables['F'] = Forcing
    problem.parameters['mu']     = mu
    problem.parameters['sigma']  = sigma
    problem.parameters['deltat'] = dt

# ── Closures (unchanged from original) ────────────────────────────────────────
problem.substitutions['mean(A)']   = "integ(A,'z')/Lz"
problem.substitutions['A(q0)']     = "a*((q0**2 + eta**2)**(1/2) - eta)"
problem.substitutions['eps(q0)']   = "c1*(Re/67)**(-1.0)*q0"
problem.substitutions['nu_zt(q0)'] = "C_zt*q0"

# q1 quasi-static approximation.
# The denominator contains the q1 decay rate. The 2*alpha term in the original
# represents the Ekman drag contribution to q1 decay. For linear-cubic drag
# this becomes 2*alpha * (1.0 + 3.0*c_lc * ulam**2) / norm_lc — the effective drag coefficient at the mean flow.
problem.substitutions['q1(w1,q0)'] = \
    "(-(wlam+w1)*dz(q0)) / (beta**2/Re + 2*kappa + 2*alpha * (1.0 + 3.0*c_lc * ulam**2) / norm_lc)"

# ══════════════════════════════════════════════════════════════════════════════
# EQUATIONS OF MOTION
# ══════════════════════════════════════════════════════════════════════════════

# ── u0 equation: bulk (y-uniform) along-band momentum ─────────────────────────
# ORIGINAL:  dt(u0) + 3*alpha*u0 - ...  = RHS
# CHANGE:    3*alpha  →  3.0 * alpha / norm_lc  = 3*alpha/norm_lc  (LHS, linear)
#            Add:     - 3.0 * alpha * c_lc / norm_lc*u0**3                 (RHS, cubic)
#
# The bulk mode has no laminar component so there are NO quadratic cross-terms.
# Just the rescaled linear coefficient and a pure cubic self-term.
# The cubic term is moved to the RHS and negated (Dedalus convention: LHS terms
# are what you write positive on the left, RHS is what drives them).
problem.add_equation(
    "dt(u0) + 3.0 * alpha / norm_lc*u0 - dz(dz(u0))/Re"
    " + S*wlam*dz(u1) + (1-S)*v1*ulam*beta"
    " = -(1-S)*v1*u1*beta - S*w1*dz(u1)"
    " - 3.0 * alpha * c_lc / norm_lc*u0**3"
    # The cubic drag on u0 is -3.0 * alpha * c_lc / norm_lc*u0^3 (opposes motion).
    # It's on the RHS because it's nonlinear and treated explicitly.
)

# ── u1 equation: shear (sin(beta*y)) along-band momentum ──────────────────────
# ORIGINAL:  dt(u1) + alpha*u1 - ...  = RHS
# CHANGE:    alpha  →  alpha * (1.0 + 3.0*c_lc * ulam**2) / norm_lc  (LHS, linear part of cubic expansion)
#            Add:   - 3.0 * alpha * c_lc * ulam / norm_lc*u1**2   (RHS, quadratic cross-term from 3*ulam*u1^2)
#            Add:   - alpha * c_lc / norm_lc*u1**3 (RHS, pure cubic term u1^3)
#
# Why these terms?  Total along-band shear = ulam + u1.
# Expanding (ulam + u1)^3 - ulam^3:
#   3*ulam^2*u1  →  absorbed into LHS as alpha * (1.0 + 3.0*c_lc * ulam**2) / norm_lc
#   3*ulam*u1^2  →  goes to RHS as -3.0 * alpha * c_lc * ulam / norm_lc*u1^2
#   u1^3         →  goes to RHS as -alpha * c_lc / norm_lc*u1^3
problem.add_equation(
    "dt(u1) + alpha * (1.0 + 3.0*c_lc * ulam**2) / norm_lc*u1 - dz(dz(u1))/Re + beta**2*u1/Re + wlam*dz(u0)"
    " = -A(q0)*beta*cos(theta) - w1*dz(u0)"
    " - 3.0 * alpha * c_lc * ulam / norm_lc*u1**2"
    # Quadratic cross-term: 3*alpha*c_lc*ulam/norm_lc * u1^2.
    # This arises because the laminar flow ulam is finite, so cubic drag
    # on (ulam+u1) produces a term proportional to ulam*u1^2.
    " - alpha * c_lc / norm_lc*u1**3"
    # Pure cubic: alpha*c_lc/norm_lc * u1^3.  Smallest of the three terms.
)

# ── zeta equation: shear vorticity (encodes w1 and v1 dynamics) ───────────────
# ORIGINAL:  dt(zeta) + alpha*beta*w1 - ...  = RHS
# CHANGE:    alpha  →  alpha * (1.0 + 3.0*c_lc * wlam**2) / norm_lc  (LHS, linear part for the w1 mode)
#            Add:   - 3.0 * alpha * c_lc * wlam / norm_lc*beta*w1**2    (RHS, quadratic cross-term)
#            Add:   - alpha * c_lc / norm_lc*beta*w1**3  (RHS, pure cubic)
#
# The across-band shear velocity is wlam + w1.
# Same expansion as u1 but with wlam replacing ulam.
# The beta factor multiplies the drag terms because zeta = beta*w1 - dz(v1).
problem.add_equation(
    "dt(zeta) + alpha * (1.0 + 3.0*c_lc * wlam**2) / norm_lc*beta*w1 - dz(dz(zeta))/Re + beta**2*zeta/Re"
    " = -beta**2*A(q0)*sin(theta) - dz(dz(A(q0)))*sin(theta)"
    " - 3.0 * alpha * c_lc * wlam / norm_lc*beta*w1**2"
    # Quadratic cross-term: 3*alpha*c_lc*wlam/norm_lc * beta * w1^2
    " - alpha * c_lc / norm_lc*beta*w1**3"
    # Pure cubic: alpha*c_lc/norm_lc * beta * w1^3
)

# ── Vorticity definition and incompressibility (unchanged) ─────────────────────
problem.add_equation("zeta - (beta*w1 - dz(v1)) = 0")
problem.add_equation("-beta*v1 + dz(w1) = 0")

# ── q0 equation: turbulent kinetic energy ─────────────────────────────────────
# ORIGINAL:  ... - 2*alpha*q0 - ...
# CHANGE:    2*alpha  →  2*alpha * (1.0 + 3.0*c_lc * ulam**2) / norm_lc
#
# For linear-cubic drag the TKE equation drag -2*alpha*q comes from
# Reynolds-averaging -alpha*h(u)*u' over fluctuations. For nonlinear drag
# this gives approximately -2*alpha_eff*q where alpha_eff is evaluated at
# the mean flow. We use alpha * (1.0 + 3.0*c_lc * ulam**2) / norm_lc = alpha * (1.0 + 3.0*c_lc * ulam**2) / norm_lc as the representative value.
# All production terms and closures are UNCHANGED — they depend on A(q0) which
# models the Reynolds stress and does not depend on the drag formulation.
if rand_force:
    problem.add_equation(
        "dt(q0) - dz(dz(q0))/Re"
        " = dz(nu_zt(q0)*dz(q0))"
        " + beta*A(q0)*cos(theta)*ulam/2"
        " + beta*A(q0)*sin(theta)*wlam/2"
        " + F(q0,mu,sigma,deltat)"
        " + beta*A(q0)*cos(theta)*u1/2"
        " + beta*A(q0)*sin(theta)*w1/2"
        " + A(q0)*sin(theta)*dz(v1)/2"
        " - 2*alpha * (1.0 + 3.0*c_lc * ulam**2) / norm_lc*q0"   # CHANGE: alpha → alpha * (1.0 + 3.0*c_lc * ulam**2) / norm_lc
        " - eps(q0)"
        " - v1*q1(w1,q0)*beta/2"
        " - (wlam+w1)*dz(q1(w1,q0))/2"
    )
    solver = problem.build_solver(de.timesteppers.RK111)
else:
    problem.add_equation(
        "dt(q0) - dz(dz(q0))/Re"
        " = dz(nu_zt(q0)*dz(q0))"
        " + beta*A(q0)*cos(theta)*ulam/2"
        " + beta*A(q0)*sin(theta)*wlam/2"
        " + beta*A(q0)*cos(theta)*u1/2"
        " + beta*A(q0)*sin(theta)*w1/2"
        " + A(q0)*sin(theta)*dz(v1)/2"
        " - 2*alpha * (1.0 + 3.0*c_lc * ulam**2) / norm_lc*q0"   # CHANGE: alpha → alpha * (1.0 + 3.0*c_lc * ulam**2) / norm_lc
        " - eps(q0)"
        " - v1*q1(w1,q0)*beta/2"
        " - (wlam+w1)*dz(q1(w1,q0))/2"
    )
    solver = problem.build_solver(de.timesteppers.RK222)

logger.info('Solver built')

# ══════════════════════════════════════════════════════════════════════════════
# INITIAL CONDITIONS OR RESTART
# ══════════════════════════════════════════════════════════════════════════════
if not pathlib.Path('restart.h5').exists():

    u0   = solver.state['u0']
    u1   = solver.state['u1']
    zeta = solver.state['zeta']
    q0   = solver.state['q0']

    cond = (k2 != 0) & (kz < (2*nIC*np.pi/Lz))

    local_coeff_shape = u0['c'].shape

    phase = np.random.uniform(low=-np.pi, high=np.pi, size=local_coeff_shape)
    u0['c'][:] = 0.0
    u0['c'][cond] = E0 * (np.cos(phase[cond]) + 1j*np.sin(phase[cond]))

    phase = np.random.uniform(low=-np.pi, high=np.pi, size=local_coeff_shape)
    u1['c'][:] = 0.0
    u1['c'][cond] = E0 * (np.cos(phase[cond]) + 1j*np.sin(phase[cond]))

    phase = np.random.uniform(low=-np.pi, high=np.pi, size=local_coeff_shape)
    zeta['c'][:] = 0.0
    zeta['c'][cond] = E0 * (np.cos(phase[cond]) + 1j*np.sin(phase[cond]))

#     if IC_file==None:
#         phase = np.random.uniform(low=-np.pi, high=np.pi, size=local_coeff_shape)
#         q0['c'][:] = 0.0
#         q0['c'][cond] = np.cos(phase[cond]) + 1j*np.sin(phase[cond])
#         q0['g'] += np.abs(np.min(q0['g'])) + 1e-4
#         q0['g'] *= Eq0 / np.max(q0['g'])
#     else:
#         q0.set_scales(2.)    # CHANGE: was 3/2, must match the new dedalus factor
#         q_in    = np.loadtxt(IC_file)
#         ind_max = np.argmin(np.abs(q_in - np.max(q_in)))
#         q_in    = np.roll(q_in, -int(ind_max - len(q_in)/2))
#         q0['g'][:] = 0.0
#         q0['g'][:len(q_in)] = q_in

#     fh_mode = 'overwrite'

# else:
#     write, last_dt  = solver.load_state('restart.h5', -1)
#     solver.sim_time  = 0.0
#     solver.iteration = 1
#     fh_mode = 'append'

# solver.stop_sim_time  = sim_tmax
# solver.stop_wall_time = real_tmax

    
    fh_mode = 'overwrite'
    
else:
    # Restart
    write, last_dt = solver.load_state('restart.h5', -1)

    ##solver.sim_time = 0.0
    ##solver.iteration = 1

    # Timestepping and output
#     dt = last_dt
    fh_mode = 'append'

# Integration parameters
solver.stop_sim_time = sim_tmax
solver.stop_wall_time = real_tmax

############
# Analysis #
############
# SNAPSHOTS
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=tsnap_sim, wall_dt = tsnap_wall, max_writes=1000, mode=fh_mode)
snapshots.add_system(solver.state)
snapshots.add_task("q1(w1,q0)",name='q1')

# TIME SERIES
t_series = solver.evaluator.add_file_handler('time_series', iter=tseries,mode=fh_mode)
t_series.add_task("mean(u0*u0+ S*u1*u1 + (1-S)*v1*v1 + S*w1*w1)/2", name='en_ls')
t_series.add_task("mean(q0)", name='q0')
t_series.add_task("sqrt(mean(q1(w1,q0)**2))", name='q1')

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("(u0*u0+ S*u1*u1 + (1-S)*v1*v1 + S*w1*w1)/2", name='KE')
flow.add_property("q0", name='KE_q')

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════
try:
    logger.info('Starting loop')
    start_time = time.time()
    while (solver.proceed):
#         dt = CFL.compute_dt()        
        solver.step(dt)
        
        if (solver.iteration-1) % 5000 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('average KE = %e' %flow.volume_average('KE'))
            logger.info('average KE_q = %e' %flow.volume_average('KE_q'))
            q_avg = flow.volume_average('KE_q')
            if (q_avg<1e-7):
                logger.error('RELAMINARIZED. Ending run.')
                raise
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.evaluate_handlers_now(dt)
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))

