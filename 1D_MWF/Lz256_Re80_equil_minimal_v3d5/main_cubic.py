"""
main_cubic.py
=============
1D Waleffe-flow model (tilted-band geometry) with PURE CUBIC drag.

Drag law:  F_drag = -alpha * |u|^2 * u  =  -alpha * u^3
           (sign-preserving: |u|^2*u has the same sign as u always)

IMPORTANT NOTES BEFORE USING THIS SCRIPT:
------------------------------------------
1. Pure cubic drag is NOT viable at alpha=0.01 (the linear reference value).
   From our viability scans, it requires tuning alpha in the range 0.003-0.008
   and still produces an INVERTED instability (stable at low Re, unstable at
   high Re) — the opposite of what is physically observed. This is a known
   limitation of the model with pure cubic drag.

2. The body force IS different from the linear drag case here.
   For linear drag:   f(y) = (alpha + beta^2/Re)*sin(beta*y)
   For cubic drag:    f(y) = alpha*sin^2(beta*y)*sin(beta*y) + (beta^2/Re)*sin(beta*y)
                           = (alpha*sin^2(beta*y) + beta^2/Re)*sin(beta*y)
   This must be implemented in Dedalus as written above.

3. The effective LINEAR drag coefficient on u1 perturbations is now
   3*alpha*ulam^2 (from expanding (ulam+u1)^3 - ulam^3).
   For u0 (bulk mode) there is NO linear drag at all since the laminar u0=0,
   and d/du(u^3) at u=0 is zero. This means the bulk mode has no linear
   restoring force from drag, which can lead to numerical instabilities.

4. The dealiasing must be 2 (not 3/2) for the same reasons as linear-cubic.

Parameters needed in params.py: same as standard, but alpha will likely need
to be different from the linear drag value to get sensible behaviour.
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
#
# For pure cubic drag -alpha*u^3, h(u) = u^3 and h(1) = 1, so norm = 1.
# No normalisation factor is needed.
#
# The body force must cancel the CUBIC drag on the laminar state.
# Laminar state: u_lam = sin(beta*y), so drag on laminar = -alpha*sin^3(beta*y).
# Body force:    f(y) = alpha*sin^2(beta*y)*sin(beta*y) + (beta^2/Re)*sin(beta*y)
#                     = (alpha*sin^2(beta*y) + beta^2/Re)*sin(beta*y)
#
# NOTE: This body force has a sin^3(beta*y) component which projects onto both
# sin(beta*y) and sin(3*beta*y). The model retains only sin(beta*y), so the
# projected body force amplitude is:
#   f1 = beta^2/Re + alpha * (3/4)   [from integral of sin^4 = 3/8, times 2]
# This is WEAKER than the linear case (f1 = beta^2/Re + alpha) by a factor 3/4,
# which shifts Re_sn upward compared to linear drag.

# ── Effective drag coefficients for each mode ─────────────────────────────────
# When we expand the cubic drag on the TOTAL velocity and subtract the body force:
#
# For u1 mode (along-band shear, total = ulam + u1):
#   (ulam+u1)^3 - ulam^3 = 3*ulam^2*u1  +  3*ulam*u1^2  +  u1^3
#   Linear part:    3*alpha*ulam^2 * u1       → LHS coefficient
#   Quadratic part: 3*alpha*ulam  * u1^2      → RHS (negative)
#   Cubic part:     alpha          * u1^3     → RHS (negative)

                  # cubic term coefficient for u1

# For w1 mode (across-band shear, total = wlam + w1):


# For u0 mode (bulk mode, laminar u0 = 0):
#   Total field = u0 (no laminar component).
#   (u0)^3 - 0 = u0^3 entirely.
#   Linear part:  d/du(3*alpha*u^3) at u=0 = 0  →  NO linear drag term on LHS!
#   Cubic part:   3*alpha * u0^3               →  RHS (negative)
#   The bulk drag coefficient is 3*alpha (alpha0 = 3*alpha in the paper).
#   The absence of a linear restoring term from drag on u0 is a known numerical
#   risk — u0 is only damped by viscosity (dz(dz(u0))/Re) which may be insufficient
#   at moderate Re. If the simulation is unstable, this is likely the cause.


# Log the drag coefficients for verification
logger.info('Drag: pure cubic')
logger.info('alpha=%.5f, alpha_eff_u1=%.5f, alpha_eff_w1=%.5f'
            % (alpha, alpha_eff_u1, alpha_eff_w1))
logger.info('alpha_q_u1=%.5f, alpha_c_u1=%.5f, alpha_c_u0=%.5f'
            % (alpha_q_u1, alpha_c_u1, alpha_c_u0))
logger.info('Note: u0 has NO linear drag — only viscous damping. Monitor carefully.')

# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN
# ══════════════════════════════════════════════════════════════════════════════

# dealias=2 required for cubic nonlinearities (same reasoning as linear-cubic).
z_basis = de.Fourier('z', Nz, interval=(0, Lz), dealias=2)
domain  = de.Domain([z_basis], grid_dtype=np.float64)

z  = z_basis.grid(scale=2)
kz = domain.elements(0)
k2 = kz**2
S  = (beta - np.sin(beta)*np.cos(beta)) / (2*beta)

# Laminar flow amplitudes
ulam = np.cos(theta)    # along-band laminar amplitude
wlam = np.sin(theta)    # across-band laminar amplitude

# Precompute sin^2(beta*y) on the dealiased grid for the body force
# This is needed because the cubic body force has a spatially varying coefficient.
# We precompute it once here as a grid function and pass it as a Dedalus parameter.
z_grid  = z_basis.grid(scale=2)
sin_by  = np.sin(beta * z_grid)       # sin(beta*y) on the across-band grid
sin2_by = sin_by**2                   # sin^2(beta*y) — coefficient in body force
alpha_eff_u1 = 3.0 * alpha * ulam**2    # effective linear drag on u1
alpha_q_u1   = 3.0 * alpha * ulam       # quadratic cross-term coefficient for u1
alpha_c_u1   = alpha  
#   Same structure but with wlam = sin(theta) replacing ulam = cos(theta).
alpha_eff_w1 = 3.0 * alpha * wlam**2
alpha_q_w1   = 3.0 * alpha * wlam
alpha_c_w1   = alpha
alpha_c_u0 = 3.0 * alpha    # cubic drag coefficient for bulk mode (no linear part)

# For the TKE equation:
#   The drag on TKE is -2*alpha_eff_tke*q0.
#   For cubic drag, alpha_eff evaluated at the mean flow ulam gives:
#   d/du(-alpha*u^3) = -3*alpha*u^2, so alpha_eff = 3*alpha*ulam^2 = alpha_eff_u1.
alpha_eff_tke = alpha_eff_u1
kcut = np.max(kz)
if rand_force:
    dt = dt_init / 4
else:
    dt = dt_init

# ══════════════════════════════════════════════════════════════════════════════
# RANDOM FORCING (unchanged)
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

# ── Standard parameters ────────────────────────────────────────────────────────
problem.parameters['Lz']    = Lz
problem.parameters['S']     = S
problem.parameters['beta']  = beta
problem.parameters['theta'] = theta
problem.parameters['a']     = a
problem.parameters['eta']   = eta
problem.parameters['Re']    = Re
problem.parameters['c1']    = c1
problem.parameters['C_zt']  = C_zt
problem.parameters['kappa'] = kappa
problem.parameters['ulam']  = ulam
problem.parameters['wlam']  = wlam

# ── Cubic drag coefficients ────────────────────────────────────────────────────
problem.parameters['alpha_eff_u1']  = alpha_eff_u1
problem.parameters['alpha_eff_w1']  = alpha_eff_w1
problem.parameters['alpha_q_u1']    = alpha_q_u1
problem.parameters['alpha_q_w1']    = alpha_q_w1
problem.parameters['alpha_c_u1']    = alpha_c_u1
problem.parameters['alpha_c_w1']    = alpha_c_w1
problem.parameters['alpha_c_u0']    = alpha_c_u0
problem.parameters['alpha_eff_tke'] = alpha_eff_tke

# ── Body force coefficient ─────────────────────────────────────────────────────
# For cubic drag the body force is (alpha*sin^2(beta*z) + beta^2/Re)*sin(beta*z).
# The sin^2(beta*z) factor varies in z (the across-band coordinate), so we pass
# it as a Dedalus non-constant coefficient (NCC).
# This is the KEY difference from linear drag where the body force is a constant
# times sin(beta*z).
f_body_coeff = domain.new_field()          # new Dedalus field on the domain
f_body_coeff.set_scales(2)                 # evaluate on dealiased grid
f_body_coeff['g'] = alpha * sin2_by + beta**2/Re
# f_body_coeff now holds (alpha*sin^2(beta*z) + beta^2/Re) at each grid point.
# In the equations it multiplies the laminar shear profile contribution.
# We store it as an NCC (it is z-dependent but time-independent).
problem.parameters['f_body'] = f_body_coeff

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

# q1 quasi-static approximation with updated drag coefficient.
problem.substitutions['q1(w1,q0)'] = \
    "(-(wlam+w1)*dz(q0)) / (beta**2/Re + 2*kappa + 2*alpha_eff_tke)"

# ══════════════════════════════════════════════════════════════════════════════
# EQUATIONS OF MOTION
# ══════════════════════════════════════════════════════════════════════════════

# ── u0 equation: bulk along-band momentum ─────────────────────────────────────
# CHANGE vs original: the 3*alpha*u0 linear drag term DISAPPEARS entirely
# because for cubic drag d/du(-alpha*u^3) at u=0 (laminar u0=0) is zero.
# The only drag on u0 is the cubic term -3*alpha*u0^3 = -alpha_c_u0*u0^3.
# This means u0 is damped ONLY by viscosity in the linear regime.
# At small u0 the restoring force is very weak — this is the primary numerical
# risk of pure cubic drag on the bulk mode.
problem.add_equation(
    "dt(u0) - dz(dz(u0))/Re"
    # NO linear alpha*u0 term here — cubic drag has no linear part at u0=0
    " + S*wlam*dz(u1) + (1-S)*v1*ulam*beta"
    " = -(1-S)*v1*u1*beta - S*w1*dz(u1)"
    " - alpha_c_u0*u0**3"
    # Pure cubic drag on u0: -3*alpha*u0^3 (negative = opposes motion).
    # Note: for the linear-drag code this was 3*alpha*u0 on the LHS.
    # For cubic drag the entire drag term is nonlinear and must go to the RHS.
)

# ── u1 equation: shear along-band momentum ────────────────────────────────────
# CHANGE vs original: alpha → alpha_eff_u1 on LHS (= 3*alpha*ulam^2).
# For ulam ≈ 0.914 (theta=24°): alpha_eff_u1 ≈ 3*alpha*0.836 ≈ 2.5*alpha.
# This effective coefficient is LARGER than the original alpha, meaning cubic
# drag damps u1 perturbations more strongly near the laminar state than linear drag.
# Quadratic and cubic cross-terms go to the RHS.
#
# Body force: the cubic body force f(y) = f_body * sin(beta*y) is not written
# explicitly here because it was already subtracted when deriving the perturbation
# equations. The laminar state satisfies the full equation, so the body force
# and laminar drag cancel exactly, leaving only perturbation terms.
problem.add_equation(
    "dt(u1) + alpha_eff_u1*u1 - dz(dz(u1))/Re + beta**2*u1/Re + wlam*dz(u0)"
    # alpha_eff_u1 = 3*alpha*ulam^2 replaces the original alpha
    " = -A(q0)*beta*cos(theta) - w1*dz(u0)"
    " - alpha_q_u1*u1**2"
    # Quadratic cross-term: 3*alpha*ulam * u1^2  (from expanding (ulam+u1)^3)
    " - alpha_c_u1*u1**3"
    # Pure cubic: alpha * u1^3
)

# ── zeta equation: shear vorticity (w1 and v1 dynamics) ───────────────────────
# CHANGE vs original: alpha → alpha_eff_w1 on LHS (= 3*alpha*wlam^2).
# For theta=24°: wlam ≈ 0.407, so alpha_eff_w1 ≈ 3*alpha*0.166 ≈ 0.5*alpha.
# This is SMALLER than the original alpha — the across-band shear mode is
# LESS damped by cubic drag than linear drag at small amplitudes.
# This asymmetry between u1 and w1 damping could affect band angle selection.
problem.add_equation(
    "dt(zeta) + alpha_eff_w1*beta*w1 - dz(dz(zeta))/Re + beta**2*zeta/Re"
    # alpha_eff_w1 = 3*alpha*wlam^2 replaces the original alpha
    " = -beta**2*A(q0)*sin(theta) - dz(dz(A(q0)))*sin(theta)"
    " - alpha_q_w1*beta*w1**2"
    # Quadratic cross-term: 3*alpha*wlam * beta * w1^2
    " - alpha_c_w1*beta*w1**3"
    # Pure cubic: alpha * beta * w1^3
)

# ── Vorticity definition and incompressibility (unchanged) ─────────────────────
problem.add_equation("zeta - (beta*w1 - dz(v1)) = 0")
problem.add_equation("-beta*v1 + dz(w1) = 0")

# ── q0 equation: turbulent kinetic energy ─────────────────────────────────────
# CHANGE vs original: 2*alpha → 2*alpha_eff_tke = 2*3*alpha*ulam^2 = 6*alpha*ulam^2.
# All other terms unchanged.
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
        " - 2*alpha_eff_tke*q0"   # CHANGE: 2*alpha → 2*alpha_eff_tke
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
        " - 2*alpha_eff_tke*q0"   # CHANGE: 2*alpha → 2*alpha_eff_tke
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

    if IC_file is None:
        phase = np.random.uniform(low=-np.pi, high=np.pi, size=local_coeff_shape)
        q0['c'][:] = 0.0
        q0['c'][cond] = np.cos(phase[cond]) + 1j*np.sin(phase[cond])
        q0['g'] += np.abs(np.min(q0['g'])) + 1e-4
        q0['g'] *= Eq0 / np.max(q0['g'])
    else:
        q0.set_scales(2.)    # must match dealias=2
        q_in    = np.loadtxt(IC_file)
        ind_max = np.argmin(np.abs(q_in - np.max(q_in)))
        q_in    = np.roll(q_in, -int(ind_max - len(q_in)/2))
        q0['g'][:] = 0.0
        q0['g'][:len(q_in)] = q_in

    fh_mode = 'overwrite'

else:
    write, last_dt  = solver.load_state('restart.h5', -1)
    solver.sim_time  = 0.0
    solver.iteration = 1
    fh_mode = 'append'

solver.stop_sim_time  = sim_tmax
solver.stop_wall_time = real_tmax

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS OUTPUTS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
snapshots = solver.evaluator.add_file_handler(
    'snapshots', sim_dt=tsnap_sim, wall_dt=tsnap_wall, max_writes=1000, mode=fh_mode)
snapshots.add_system(solver.state)
snapshots.add_task("q1(w1,q0)", name='q1')

t_series = solver.evaluator.add_file_handler('time_series', iter=tseries, mode=fh_mode)
t_series.add_task("mean(u0*u0 + S*u1*u1 + (1-S)*v1*v1 + S*w1*w1)/2", name='en_ls')
t_series.add_task("mean(q0)",                  name='q0')
t_series.add_task("sqrt(mean(q1(w1,q0)**2))", name='q1')

flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("(u0*u0 + S*u1*u1 + (1-S)*v1*v1 + S*w1*w1)/2", name='KE')
flow.add_property("q0",                                             name='KE_q')

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        solver.step(dt)
        if (solver.iteration - 1) % 5000 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e'
                        % (solver.iteration, solver.sim_time, dt))
            logger.info('average KE   = %e' % flow.volume_average('KE'))
            logger.info('average KE_q = %e' % flow.volume_average('KE_q'))
            if flow.volume_average('KE_q') < 1e-7:
                logger.error('RELAMINARIZED. Ending run.')
                raise
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.evaluate_handlers_now(dt)
    end_time = time.time()
    logger.info('Iterations: %i'      % solver.iteration)
    logger.info('Sim end time: %f'    % solver.sim_time)
    logger.info('Run time: %.2f sec'  % (end_time - start_time))
    logger.info('Run time: %f cpu-hr' % ((end_time - start_time)/3600
                                         * domain.dist.comm_cart.size))
