import numpy as np
import h5py
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import time
import pathlib
from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

ncores = comm.Get_size()
nnodes = int(ncores/16)
logger.info('Running on %s cores,  %s nodes' % (ncores,nnodes))

import scipy.special as spec

from params import *

logger.info('Lx = %.7f, Lz = %.7f, Re = %.4e' % (Lx,Lz,Re))

##########
# DOMAIN #
##########

# Create bases and domain
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
z_basis = de.Fourier('z', Nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([x_basis,z_basis], grid_dtype=np.float64)

# For general use
x = domain.grid(0)
z = domain.grid(1)
kx = domain.elements(0)
kz = domain.elements(1)
k2 = kx**2+kz**2
S = (beta - np.sin(beta)*np.cos(beta))/(2*beta)

# Timestepping and output
if rand_force:
    dt = dt_init/4
else:
    dt = dt_init

##################
# Random forcing #
##################
if rand_force:
    def my_forcing(*args):
        q = args[0].data
        mu = args[1].value
        sigma = args[2].value
        deltat = args[3].value
        return np.sqrt(deltat)*q*np.random.normal(mu,sigma,size=q.shape)
    def Forcing(*args, domain=domain, F=my_forcing):
        return de.operators.GeneralFunction(domain, layout='g', func=F, args=args)

#############
# EQUATIONS #
#############
# 2D MWF
problem = de.IVP(domain, variables=['u0','w0','u1','v1','w1','p0','p1','q0'])
problem.parameters['Lx'] = Lx
problem.parameters['Lz'] = Lz
problem.parameters['S'] = S
problem.parameters['beta'] = beta
problem.parameters['a'] = a
problem.parameters['b'] = b
problem.parameters['eta'] = eta
# Dissipation
problem.parameters['Re'] = Re
problem.parameters['c1'] = c1
problem.parameters['kappa'] = kappa
problem.parameters['C_zt'] = C_zt
problem.parameters['alpha'] = alpha
problem.substitutions['Lap(a)'] = "dx(dx(a)) + dz(dz(a))"
if rand_force:
    # Forcing
    de.operators.parseables['F'] = Forcing
    problem.parameters['mu'] = mu 
    problem.parameters['sigma'] = sigma 
    problem.parameters['deltat'] = dt 
# Closures
problem.substitutions['A(q0)'] = "a*((q0**2 + eta**2)**(1/2) - eta)"
problem.substitutions['B(q0)'] = "b*abs(q0)"
problem.substitutions['eps(q0)'] = "c1*(Re/67)**(-1.0)*q0"
problem.substitutions['nu_zt(q0)'] = "C_zt*q0"
problem.substitutions['q1(u1,w1,q0)'] = "(-dx(q0) - u1*dx(q0) -w1*dz(q0))/(beta**2/Re + 2*alpha + 2*kappa)"
problem.substitutions['mean(a)'] = "integ(integ(a,'z'),'x')/Lz/Lx"
# Equtions of motion
# Large scales
# zero mode
problem.add_equation("dt(u0) + S*(dx(u1) + beta*v1) + dx(p0) - Lap(u0)/Re + 3*alpha*u0 = - u0*dx(u0) - w0*dz(u0) - S*(u1*dx(u1) + beta*v1*u1 + w1*dz(u1))-dx(B(q0))")
problem.add_equation("dt(w0) + S*dx(w1)             + dz(p0) - Lap(w0)/Re + 3*alpha*w0 = - u0*dx(w0) - w0*dz(w0) - S*(u1*dx(w1) + beta*v1*w1 + w1*dz(w1))")
problem.add_equation("dx(u0) + dz(w0) = 0", condition="(nx != 0) or (nz != 0)")
problem.add_equation("integ(p0) = 0", condition="(nx == 0) and (nz == 0)") # Pressure gauge 0 
# first mode
problem.add_equation("dt(u1) + dx(u0) + dx(p1) - Lap(u1)/Re + (alpha+beta**2/Re)*u1 = - beta*A(q0) -u0*dx(u1) - u1*dx(u0) - w0*dz(u1) - w1*dz(u0)")
problem.add_equation("dt(w1) + dx(w0) + dz(p1) - Lap(w1)/Re + (alpha+beta**2/Re)*w1 =              -u0*dx(w1) - u1*dx(w0) - w0*dz(w1) - w1*dz(w0)")
problem.add_equation("dt(v1) + beta*p1 - Lap(v1)/Re + beta**2*v1/Re = dx(A(q0)) -u0*dx(v1) - w0*dz(v1)")
problem.add_equation("dx(u1) - beta*v1 + dz(w1) = 0")
# Turbulence
if rand_force:
    #q0
    problem.add_equation("dt(q0) - Lap(q0)/Re + 2*alpha*q0 = -u0*dx(q0) - w0*dz(q0) + S*A(q0)*(beta*(1+u1)+dx(v1)) - eps(q0) + dx(nu_zt(q0)*dx(q0)) + dz(nu_zt(q0)*dz(q0)) + F(q0,mu,sigma,deltat) -S*dx(q1(u1,w1,q0)) - S*u1*dx(q1(u1,w1,q0)) - S*beta*v1*q1(u1,w1,q0) - S*w1*dz(q1(u1,w1,q0))-B(q0)*dx(u0)")
    
    # Build solver (first order due to random forcing)
    solver = problem.build_solver(de.timesteppers.RK111)
else:
    #q0
    problem.add_equation("dt(q0) - Lap(q0)/Re + 2*alpha*q0= -u0*dx(q0) - w0*dz(q0) + S*A(q0)*(beta*(1+u1)+dx(v1)) - eps(q0) + dx(nu_zt(q0)*dx(q0)) + dz(nu_zt(q0)*dz(q0)) -S*dx(q1(u1,w1,q0)) - S*u1*dx(q1(u1,w1,q0)) - S*beta*v1*q1(u1,w1,q0) - S*w1*dz(q1(u1,w1,q0)) -B(q0)*dx(u0)")
    
    # Build solver
    solver = problem.build_solver(de.timesteppers.RK222)
    #solver = problem.build_solver(de.timesteppers.RK443)

logger.info('Solver built')

#################################
# Initial conditions or restart #
#################################
if not pathlib.Path('restart.h5').exists():

    # Initial conditions
    u0 = solver.state['u0']
    u1 = solver.state['u1']
    v1 = solver.state['v1']
    w0 = solver.state['w0']
    w1 = solver.state['w1']
    q0 = solver.state['q0']
    
    # Random initial conditions for k <= kf_max, otherwise = 0
    cond = (k2!=0)&(np.abs(kz)<(2*nICz*np.pi/Lz))&(np.abs(kx)<(2*nICx*np.pi/Lx))
    
    local_coeff_shape=u0['c'].shape
    phase = np.random.uniform(low=-np.pi,high=np.pi,size=local_coeff_shape)
    u0['c'][:]=0.0
    u0['c'][cond] = E0*(np.cos(phase[cond])+1j*np.sin(phase[cond]))
   
    phase = np.random.uniform(low=-np.pi,high=np.pi,size=local_coeff_shape)
    u1['c'][:]=0.0
    u1['c'][cond] = E0*(np.cos(phase[cond])+1j*np.sin(phase[cond]))
    
    phase = np.random.uniform(low=-np.pi,high=np.pi,size=local_coeff_shape)
    v1['c'][:]=0.0
    v1['c'][cond] = E0*(np.cos(phase[cond])+1j*np.sin(phase[cond]))
    
    phase = np.random.uniform(low=-np.pi,high=np.pi,size=local_coeff_shape)
    w0['c'][:]=0.0
    w0['c'][cond] = E0*(np.cos(phase[cond])+1j*np.sin(phase[cond]))
   
    phase = np.random.uniform(low=-np.pi,high=np.pi,size=local_coeff_shape)
    w1['c'][:]=0.0
    w1['c'][cond] = E0*(np.cos(phase[cond])+1j*np.sin(phase[cond]))
    
    if IC_file==None:
        phase = np.random.uniform(low=-np.pi,high=np.pi,size=local_coeff_shape)
        q0['c'][:]=0.0
        q0['c'][cond] = (np.cos(phase[cond])+1j*np.sin(phase[cond]))
        # Make sure q>0
        q0min = comm.allreduce(np.min(q0['g']),op=MPI.MIN)
        q0['g'] += np.abs(q0min)+1e-4
        # Make the max Eq0
        q0max = comm.allreduce(np.max(q0['g']),op=MPI.MAX)
        q0['g'] *= Eq0/q0max
    elif IC_file=='gauss':
        gauss_width=10
        X,Z = np.meshgrid(x,z,indexing='ij')
        gauss = Eq0*np.exp(-((X-Lx/2)**2+(Z-Lz/2)**2)/(2*gauss_width**2))
        hamming_x = 0.54 - 0.46*np.cos(2*np.pi*x/Lx)
        hamming_z = 0.54 - 0.46*np.cos(2*np.pi*z/Lz)
        HX,HZ = np.meshgrid(hamming_x,hamming_z,indexing='ij')
        q0['g'] = gauss*HX*HZ
    elif IC_file=='tube':
        gauss_width=7
        X,Z = np.meshgrid(x,z,indexing='ij')
        # Create single band
        Xcross = X*np.cos(theta_IC) + Z*np.sin(theta_IC)
        Ccross= Lx*np.cos(theta_IC)/2 + Lz*np.sin(theta_IC)/2
        gauss = Eq0*np.exp(-((Xcross-Ccross)**2)/(2*gauss_width**2))

        # Truncate length
        Xalong = -X*np.sin(theta_IC) + Z*np.cos(theta_IC)
        Calong= -Lx*np.sin(theta_IC)/2 + Lz*np.cos(theta_IC)/2
        gauss *= (1+spec.erf((-Xalong + Calong + L_IC)/gauss_width/2))/2 * (1+spec.erf((Xalong - Calong + L_IC)/gauss_width/2))/2  
        q0['g'] = gauss
    elif IC_file=='equil':
        ## Calibrated by MWF (newuq)
        def A(q,a,eta,Re):
            return a*((q**2 + eta**2)**(1/2) - eta)

        def B(q,b):
            return b*q

        def eps(q,c1,Re):
            return c1*(Re/67)**(-1)*q
        # Define nullclines (for plotting)
        def null_u(q,beta,eta,a,alpha,Re):
            return -beta*A(q,a,eta,Re)/(alpha+beta**2/Re) + 1 # u nullcline
        def null_q(q,beta,eta,a,c1,alpha,Re): 
            return (2*alpha*q + eps(q,c1,Re))/(beta*A(q,a,eta,Re)/2)
        # Find equilibrium
        qplt = np.linspace(1e-4,1,10000)
        if (Re<73):
            Re_fp = 74
        else:
            Re_fp = Re
        # Equilibria
        diff = np.abs(null_u(qplt,beta,eta,a,alpha,Re_fp)-null_q(qplt,beta,eta,a,c1,alpha,Re_fp))
        ind1 = np.argsort(diff)[0]
        ind2 = np.argsort(diff)[1]
        # Check for the largest q
        if qplt[ind1]<qplt[ind2]:
            ind = ind2
        else:
            ind = ind1
        qe = qplt[ind]
        q0['g'][:,:]=qe
        u1_se = null_u(qplt,beta,eta,a,alpha,Re_fp)[ind]-1
        u1['g'][:,:]=u1_se
    
    fh_mode = 'overwrite'
    
else:
    # Restart
    write, last_dt = solver.load_state('restart.h5', -1)

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
snapshots.add_task("q1(u1,w1,q0)",name='q1')

# TIME SERIES
t_series = solver.evaluator.add_file_handler('time_series', iter=tseries,mode=fh_mode)
t_series.add_task("mean(u0*u0+ S*u1*u1 + (1-S)*v1*v1 + w0*w0 + S*w1*w1)/2", name='en_ls')
t_series.add_task("mean(q0)", name='q0')
t_series.add_task("sqrt(mean(q1(u1,w1,q0)**2))", name='q1')

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("(u0*u0+ S*u1*u1 + (1-S)*v1*v1 + w0*w0 + S*w1*w1)/2", name='KE')
flow.add_property("q0", name='KE_q')

# Main loop
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
            if (q_avg<1e-8):
                logger.error('RELAMINARIZED. Ending run.')
                raise
except Exception as e:
    logger.error('Exception raised, triggering end of main loop.')
    logger.error(str(e))
    raise
finally:
    solver.evaluate_handlers_now(dt)
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
