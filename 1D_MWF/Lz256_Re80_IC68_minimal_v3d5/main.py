# import numpy as np
# import h5py
# from mpi4py import MPI

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# import time
# import pathlib
# from dedalus import public as de
# from dedalus.extras import flow_tools
# from scipy import interpolate

# import logging
# logger = logging.getLogger(__name__)

# ncores = comm.Get_size()
# nnodes = int(ncores/16)
# logger.info('Running on %s cores,  %s nodes' % (ncores,nnodes))

# from params import *

# logger.info('Lz = %.7f, Re = %.4e' % (Lz,Re))


# #############################
# # DEFINITIONS AND OPERATORS #
# #############################
# import spec1d
# import flux1d
# import nonlin_term
# import vector_cal as vc

# ##########
# # DOMAIN #
# ##########

# # Create bases and domain
# z_basis = de.Fourier('z', Nz, interval=(0, Lz), dealias=3/2)
# domain = de.Domain([z_basis], grid_dtype=np.float64)#,mesh=(int(Nx/2),int(2*ncores/Nx)))

# # For general use
# z = z_basis.grid(scale=3/2)
# kz = domain.elements(0)
# k2 = kz**2
# S = (beta - np.sin(beta)*np.cos(beta))/(2*beta)

# ######################
# # Laminar Flow Field #
# ######################
# ulam = np.cos(theta)
# wlam = np.sin(theta)

# # Timestepping and output
# kcut = np.max(kz) # not Nx/3 because of the 3/2 dealiasing rule instead of the 2/3 rule.
# if rand_force:
#     dt = dt_init/4
# else:
#     dt = dt_init

# ##################
# # Random forcing #
# ##################
# if rand_force:
#     def my_forcing(*args):
#         q = args[0].data
#         mu = args[1].value
#         sigma = args[2].value
#         deltat = args[3].value
#         return np.sqrt(deltat)*q*np.random.normal(mu,sigma,size=q.shape)
#     def Forcing(*args, domain=domain, F=my_forcing):
#         return de.operators.GeneralFunction(domain, layout='g', func=F, args=args)

# #############
# # EQUATIONS #
# #############
# # 1D MWF
# problem = de.IVP(domain, variables=['u0','u1','v1','w1','zeta','q0'])
# problem.parameters['Lz'] = Lz
# problem.parameters['S'] = S
# problem.parameters['beta'] = beta
# problem.parameters['theta'] = theta
# problem.parameters['a'] = a
# problem.parameters['eta'] = eta
# # Dissipation
# problem.parameters['Re'] = Re
# problem.parameters['c1'] = c1
# problem.parameters['C_zt'] = C_zt
# problem.parameters['alpha'] = alpha
# problem.parameters['kappa'] = kappa
# # Laminar solution
# problem.parameters['ulam'] = ulam
# problem.parameters['wlam'] = wlam
# if rand_force:
#     # Forcing
#     de.operators.parseables['F'] = Forcing
#     problem.parameters['mu'] = mu 
#     problem.parameters['sigma'] = sigma 
#     problem.parameters['deltat'] = dt 
# # For outputs
# problem.substitutions['mean(a)'] = "integ(a,'z')/Lz"
# # Closures
# problem.substitutions['A(q0)'] = "a*((q0**2 + eta**2)**(1/2) - eta)"
# problem.substitutions['eps(q0)'] = "c1*(Re/67)**(-1.0)*q0"
# problem.substitutions['nu_zt(q0)'] = "C_zt*q0"
# problem.substitutions['q1(w1,q0)'] = "(-(wlam+w1)*dz(q0))/(beta**2/Re + 2*kappa + 2*alpha)"
# # Equtions of motion
# # Large scales
# problem.add_equation("dt(u0) + 3*alpha*u0 - dz(dz(u0))/Re + S*wlam*dz(u1) + (1-S)*v1*ulam*beta = -(1-S)*v1*u1*beta - S*w1*dz(u1)")
# problem.add_equation("dt(u1) + alpha*u1 - dz(dz(u1))/Re + beta**2*u1/Re + wlam*dz(u0) = -A(q0)*beta*cos(theta) -w1*dz(u0)")
# problem.add_equation("dt(zeta) + alpha*beta*w1 - dz(dz(zeta))/Re + beta**2*zeta/Re = -beta**2*A(q0)*sin(theta) - dz(dz(A(q0)))*sin(theta)")
# problem.add_equation("zeta - (beta*w1-dz(v1)) = 0")
# problem.add_equation("-beta*v1 + dz(w1) = 0")
# # Turbulence
# if rand_force:
#     problem.add_equation("dt(q0) -dz(dz(q0))/Re = dz(nu_zt(q0)*dz(q0)) +  beta*A(q0)*cos(theta)*ulam/2 + beta*A(q0)*sin(theta)*wlam/2 + F(q0,mu,sigma,deltat) + beta*A(q0)*cos(theta)*u1/2 +beta*A(q0)*sin(theta)*w1/2 + A(q0)*sin(theta)*dz(v1)/2 - 2*alpha*q0 - eps(q0) -v1*q1(w1,q0)*beta/2 -(wlam+w1)*dz(q1(w1,q0))/2")
    
#     # Build solver (first order due to random forcing)
#     solver = problem.build_solver(de.timesteppers.RK111)
# else:
#     problem.add_equation("dt(q0) -dz(dz(q0))/Re = dz(nu_zt(q0)*dz(q0)) + beta*A(q0)*cos(theta)*ulam/2 + beta*A(q0)*sin(theta)*wlam/2 + beta*A(q0)*cos(theta)*u1/2 +beta*A(q0)*sin(theta)*w1/2 + A(q0)*sin(theta)*dz(v1)/2 - 2*alpha*q0 - eps(q0) -v1*q1(w1,q0)*beta/2 -(wlam+w1)*dz(q1(w1,q0))/2")    

#     # Build solver
#     ##solver = problem.build_solver(de.timesteppers.RK443)
#     solver = problem.build_solver(de.timesteppers.RK222)

# logger.info('Solver built')

# #################################
# # Initial conditions or restart #
# #################################
# if not pathlib.Path('restart.h5').exists():

#     # Initial conditions
#     u0 = solver.state['u0']
#     u1 = solver.state['u1']
#     w1 = solver.state['w1']
#     zeta = solver.state['zeta']
#     q0 = solver.state['q0']
    
#     z = z_basis.grid(scale=1)
#     q0.set_scales(1)
#     u0.set_scales(1)
#     u1.set_scales(1)
#     w1.set_scales(1)
#     zeta.set_scales(1)
    
#     # Random initial conditions for k <= kf_max, otherwise = 0
#     cond = (k2!=0)&(kz<(2*nIC*np.pi/Lz))
    
#     local_coeff_shape=u0['c'].shape
#     phase = np.random.uniform(low=-np.pi,high=np.pi,size=local_coeff_shape)
#     u0['c'][:]=0.0
#     u0['c'][cond] = E0*(np.cos(phase[cond])+1j*np.sin(phase[cond]))
   
#     phase = np.random.uniform(low=-np.pi,high=np.pi,size=local_coeff_shape)
#     u1['c'][:]=0.0
#     u1['c'][cond] = E0*(np.cos(phase[cond])+1j*np.sin(phase[cond]))
    
#     phase = np.random.uniform(low=-np.pi,high=np.pi,size=local_coeff_shape)
#     zeta['c'][:]=0.0
#     zeta['c'][cond] = E0*(np.cos(phase[cond])+1j*np.sin(phase[cond]))
    
#     if IC_file != None:
#         try:
#             [z_IC,q0_IC,u0_IC,u1_IC,w1_IC,zeta_IC,q1_IC] = np.load(IC_file,allow_pickle=True)
#         except:
#             [z_IC,q0_IC,u0_IC,u1_IC,w1_IC,zeta_IC] = np.load(IC_file,allow_pickle=True)
#         if len(z_IC)==len(z):
#             q0['g'][:] = q0_IC[:]
#             u0['g'][:] = u0_IC[:]
#             u1['g'][:] = u1_IC[:]
#             w1['g'][:] = w1_IC[:]
#             zeta['g'][:] = zeta_IC[:]
#         else:
#             #Interpolate!
#             q0['g'][:] = 0.0
#             u0['g'][:] = 0.0
#             u1['g'][:] = 0.0
#             w1['g'][:] = 0.0
#             zeta['g'][:] = 0.0
#             q0int = interpolate.interp1d(z_IC, q0_IC,kind='linear')
#             q0['g'][(z>=z_IC[0])&(z<=z_IC[-1])] = q0int(z[(z>=z_IC[0])&(z<=z_IC[-1])])
#             u0int = interpolate.interp1d(z_IC, u0_IC,kind='linear')
#             u0['g'][(z>=z_IC[0])&(z<=z_IC[-1])] = u0int(z[(z>=z_IC[0])&(z<=z_IC[-1])])
#             u1int = interpolate.interp1d(z_IC, u1_IC,kind='linear')
#             u1['g'][(z>=z_IC[0])&(z<=z_IC[-1])] = u1int(z[(z>=z_IC[0])&(z<=z_IC[-1])])
#             w1int = interpolate.interp1d(z_IC, w1_IC,kind='linear')
#             w1['g'][(z>=z_IC[0])&(z<=z_IC[-1])] = w1int(z[(z>=z_IC[0])&(z<=z_IC[-1])])
#             zetaint = interpolate.interp1d(z_IC, zeta_IC,kind='linear')
#             zeta['g'][(z>=z_IC[0])&(z<=z_IC[-1])] = zetaint(z[(z>=z_IC[0])&(z<=z_IC[-1])])
#     else: # Make a guess
#         # Find equilibrium
#         qplt = np.linspace(1e-4,1,10000)
#         ## Calibrated by MWF (newuq)
#         def A_eq(q,a,eta,Re):
#             return a*((q**2 + eta**2)**(1/2) - eta)
#         def eps_eq(q,c1,Re):
#             return c1*(Re/67)**(-1)*q
#         # Define nullclines (for plotting)
#         def null_u(q,beta,eta,a,alpha,Re):
#             return -beta*A_eq(q,a,eta,Re)/(alpha+beta**2/Re) + 1 # u nullcline
#         def null_q(q,beta,eta,a,c1,alpha,Re):
#             return (2*alpha*q + eps_eq(q,c1,Re))/(beta*A_eq(q,a,eta,Re)/2)
#         # Equilibria
#         diff = np.abs(null_u(qplt,beta,eta,a,alpha,Re)-null_q(qplt,beta,eta,a,c1,alpha,Re))
#         ind1 = np.argsort(diff)[0]
#         ind2 = np.argsort(diff)[1]
#         # Check for the largest q
#         if qplt[ind1]<qplt[ind2]:
#             ind = ind2
#         else:
#             ind = ind1
#         qe = qplt[ind]
#         u1_se = null_u(qplt,beta,eta,a,alpha,Re)[ind]
#         # Rotate:
#         u1e = u1_se*np.cos(theta)
#         w1e = u1_se*np.sin(theta)
#         # Add perturbation at largest wavenumber:
#         q0['g'][:] = qe  + q0_c + R_q0*np.cos(2*np.pi*z/Lz)
#         u1['g'][:] = u1e + u1_c + R_u1*np.cos(2*np.pi*z/Lz)
#         u0['g'][:] =  0  + u0_c + R_u0*np.sin(2*np.pi*(z-Lz/4)/Lz)
#         w1['g'][:] = w1e + w1_c + R_w1**np.cos(2*np.pi*z/Lz)
        
    
# #    if IC_file==None:
# #        phase = np.random.uniform(low=-np.pi,high=np.pi,size=local_coeff_shape)
# #        q0['c'][:]=0.0
# #        q0['c'][cond] = (np.cos(phase[cond])+1j*np.sin(phase[cond]))
# #        # Make sure q>0
# #        q0['g'] += np.abs(np.min(q0['g']))+1e-4
# #        # Make the max Eq0
# #        q0['g'] *= Eq0/np.max(q0['g'])
# #    else:
# #        q0.set_scales(3/2.)
# #        q_in = np.loadtxt(IC_file)
# #        # Center in domain:
# #        ind_max = np.argmin(np.abs(q_in-np.max(q_in)))
# #        q_in = np.roll(q_in,-int(ind_max-len(q_in)/2))
# #        q0['g'][:] = 0.0
# #        q0['g'][:len(q_in)] = q_in
    
#     fh_mode = 'overwrite'
    
# else:
#     # Restart
#     write, last_dt = solver.load_state('restart.h5', -1)

#     solver.sim_time = 0.0
#     solver.iteration = 1

#     # Timestepping and output
# #     dt = last_dt
#     fh_mode = 'append'

# # Integration parameters
# solver.stop_sim_time = sim_tmax
# solver.stop_wall_time = real_tmax

# # #######
# # # CFL #
# # #######
# # CFL = flow_tools.CFL(solver, initial_dt=dt, safety=0.5, cadence=100, threshold=0.05)
# # CFL.add_velocity('w',axis=0)
# # CFL.add_velocity('wlam',axis=0)
# # CFL.add_velocity('a*q*sin(theta)',axis=0)
# # # Momentum diffusion
# # kcut = np.max(kz)
# # CFL.add_frequency(kcut**(2)/Re)
# # CFL.add_frequency(drag)
# # CFL.add_frequency(kappa)

# # logger.info('1/dt restriction from visc = %.3e, drag = %.3e, kappa = %.3e' % (kcut**(2)/Re,drag,kappa))

# ############
# # Analysis #
# ############
# # SNAPSHOTS
# snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=tsnap_sim, wall_dt = tsnap_wall, max_writes=1000, mode=fh_mode)
# snapshots.add_system(solver.state)
# snapshots.add_task("q1(w1,q0)",name='q1')

# # TIME SERIES
# t_series = solver.evaluator.add_file_handler('time_series', iter=tseries,mode=fh_mode)
# t_series.add_task("mean(u0*u0+ S*u1*u1 + (1-S)*v1*v1 + S*w1*w1)/2", name='en_ls')
# t_series.add_task("mean(q0)", name='q0')
# t_series.add_task("sqrt(mean(q1(w1,q0)**2))", name='q1')

# # Flow properties
# flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
# flow.add_property("(u0*u0+ S*u1*u1 + (1-S)*v1*v1 + S*w1*w1)/2", name='KE')
# flow.add_property("q0", name='KE_q')

# # Main loop
# try:
#     logger.info('Starting loop')
#     start_time = time.time()
#     while (solver.proceed):
# #         dt = CFL.compute_dt()        
#         solver.step(dt)
        
#         if (solver.iteration-1) % 5000 == 0:
#             logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
#             logger.info('average KE = %e' %flow.volume_average('KE'))
#             logger.info('average KE_q = %e' %flow.volume_average('KE_q'))
#             q_avg = flow.volume_average('KE_q')
#             if (q_avg<1e-7):
#                 logger.error('RELAMINARIZED. Ending run.')
#                 raise
# except:
#     logger.error('Exception raised, triggering end of main loop.')
#     raise
# finally:
#     solver.evaluate_handlers_now(dt)
#     end_time = time.time()
#     logger.info('Iterations: %i' %solver.iteration)
#     logger.info('Sim end time: %f' %solver.sim_time)
#     logger.info('Run time: %.2f sec' %(end_time-start_time))
#     logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))



    ############################
    ###### LINEAR CUBIC #######
    ###########################


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

from params import *

logger.info('Lz = %.7f, Re = %.4e' % (Lz,Re))


#############################
# DEFINITIONS AND OPERATORS #
#############################
import spec1d
import flux1d
import nonlin_term
import vector_cal as vc

##########
# DOMAIN #
##########

# Create bases and domain
z_basis = de.Fourier('z', Nz, interval=(0, Lz), dealias=3/2)
domain = de.Domain([z_basis], grid_dtype=np.float64)#,mesh=(int(Nx/2),int(2*ncores/Nx)))

# For general use
z = z_basis.grid(scale=3/2)
kz = domain.elements(0)
k2 = kz**2
S = (beta - np.sin(beta)*np.cos(beta))/(2*beta)

######################
# Laminar Flow Field #
######################
ulam = np.cos(theta)
wlam = np.sin(theta)

# Timestepping and output
kcut = np.max(kz) # not Nx/3 because of the 3/2 dealiasing rule instead of the 2/3 rule.
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
# 1D MWF
problem = de.IVP(domain, variables=['u0','u1','v1','w1','zeta','q0'])
problem.parameters['Lz'] = Lz
problem.parameters['S'] = S
problem.parameters['beta'] = beta
problem.parameters['theta'] = theta
problem.parameters['a'] = a
problem.parameters['eta'] = eta
# Dissipation
problem.parameters['Re'] = Re
problem.parameters['c1'] = c1
problem.parameters['C_zt'] = C_zt
problem.parameters['alpha'] = alpha
problem.parameters['kappa'] = kappa
# Laminar solution
problem.parameters['ulam'] = ulam
problem.parameters['wlam'] = wlam
c_lc= .1
norm_lc = 1 + c_lc
problem.parameters['-c_lc'] = c_lc
problem.parameters['norm_lc'] = norm_lc

alpha_u1_lin = alpha * (1 + 3*c_lc * ulam**2) / norm_lc   # effective linear coeff for u1
alpha_u1_q   = 3 * alpha * c_lc * ulam / norm_lc           # quadratic cross-term coeff
alpha_u1_c   = alpha * c_lc / norm_lc                       # cubic term coeff
problem.parameters['alpha_u1_lin'] = alpha_u1_lin
problem.parameters['alpha_u1_q']   = alpha_u1_q
problem.parameters['alpha_u1_c']   = alpha_u1_c

alpha_w1_lin = alpha * (1 + 3*c_lc * wlam**2) / norm_lc
alpha_w1_q   = 3 * alpha * c_lc * wlam / norm_lc
alpha_w1_c   = alpha * c_lc / norm_lc
problem.parameters['alpha_w1_lin'] = alpha_w1_lin
problem.parameters['alpha_w1_q']   = alpha_w1_q
problem.parameters['alpha_w1_c']   = alpha_w1_c

alpha_u0_lin = 3 * alpha / norm_lc
alpha_u0_c   = 3 * alpha * c_lc / norm_lc
problem.parameters['alpha_u0_lin'] = alpha_u0_lin
problem.parameters['alpha_u0_c']   = alpha_u0_c
if rand_force:
    # Forcing
    de.operators.parseables['F'] = Forcing
    problem.parameters['mu'] = mu
    problem.parameters['sigma'] = sigma
    problem.parameters['deltat'] = dt

# For outputs
problem.substitutions['mean(a)'] = "integ(a,'z')/Lz"
# Closures
problem.substitutions['A(q0)'] = "a*((q0**2 + eta**2)**(1/2) - eta)"
problem.substitutions['eps(q0)'] = "c1*(Re/67)**(-1.0)*q0"
problem.substitutions['nu_zt(q0)'] = "C_zt*q0"
problem.substitutions['q1(w1,q0)'] = "(-(wlam+w1)*dz(q0))/(beta**2/Re + 2*kappa + 2*alpha_tke)"
# Equtions of motion
# Large scales
problem.add_equation("dt(u0) + 3*alpha*u0 - dz(dz(u0))/Re + S*wlam*dz(u1) + (1-S)*v1*ulam*beta = -(1-S)*v1*u1*beta - S*w1*dz(u1)")
problem.add_equation("dt(u1) + alpha*u1 - dz(dz(u1))/Re + beta**2*u1/Re + wlam*dz(u0) = -A(q0)*beta*cos(theta) -w1*dz(u0)")
problem.add_equation("dt(zeta) + alpha*beta*w1 - dz(dz(zeta))/Re + beta**2*zeta/Re = -beta**2*A(q0)*sin(theta) - dz(dz(A(q0)))*sin(theta)")
problem.add_equation("zeta - (beta*w1-dz(v1)) = 0")
problem.add_equation("-beta*v1 + dz(w1) = 0")
# Turbulence
if rand_force:
    problem.add_equation("dt(q0) -dz(dz(q0))/Re = dz(nu_zt(q0)*dz(q0)) +  beta*A(q0)*cos(theta)*ulam/2 + beta*A(q0)*sin(theta)*wlam/2 + F(q0,mu,sigma,deltat) + beta*A(q0)*cos(theta)*u1/2 +beta*A(q0)*sin(theta)*w1/2 + A(q0)*sin(theta)*dz(v1)/2 - 2*alpha_tke*q0 - eps(q0) -v1*q1(w1,q0)*beta/2 -(wlam+w1)*dz(q1(w1,q0))/2")
   
    # Build solver (first order due to random forcing)
    solver = problem.build_solver(de.timesteppers.RK111)
else:
    problem.add_equation("dt(q0) -dz(dz(q0))/Re = dz(nu_zt(q0)*dz(q0)) + beta*A(q0)*cos(theta)*ulam/2 + beta*A(q0)*sin(theta)*wlam/2 + beta*A(q0)*cos(theta)*u1/2 +beta*A(q0)*sin(theta)*w1/2 + A(q0)*sin(theta)*dz(v1)/2 - 2*alpha_tke10*q0 - eps(q0) -v1*q1(w1,q0)*beta/2 -(wlam+w1)*dz(q1(w1,q0))/2")    

    # Build solver
    ##solver = problem.build_solver(de.timesteppers.RK443)
    solver = problem.build_solver(de.timesteppers.RK222)

logger.info('Solver built')

#################################
# Initial conditions or restart #
#################################
if not pathlib.Path('restart.h5').exists():

    # Initial conditions
    u0 = solver.state['u0']
    u1 = solver.state['u1']
    zeta = solver.state['zeta']
    q0 = solver.state['q0']
   
    # Random initial conditions for k <= kf_max, otherwise = 0
    cond = (k2!=0)&(kz<(2*nIC*np.pi/Lz))
   
    local_coeff_shape=u0['c'].shape
    phase = np.random.uniform(low=-np.pi,high=np.pi,size=local_coeff_shape)
    u0['c'][:]=0.0
    u0['c'][cond] = E0*(np.cos(phase[cond])+1j*np.sin(phase[cond]))
   
    phase = np.random.uniform(low=-np.pi,high=np.pi,size=local_coeff_shape)
    u1['c'][:]=0.0
    u1['c'][cond] = E0*(np.cos(phase[cond])+1j*np.sin(phase[cond]))
   
    phase = np.random.uniform(low=-np.pi,high=np.pi,size=local_coeff_shape)
    zeta['c'][:]=0.0
    zeta['c'][cond] = E0*(np.cos(phase[cond])+1j*np.sin(phase[cond]))
   
    if IC_file==None:
        phase = np.random.uniform(low=-np.pi,high=np.pi,size=local_coeff_shape)
        q0['c'][:]=0.0
        q0['c'][cond] = (np.cos(phase[cond])+1j*np.sin(phase[cond]))
        # Make sure q>0
        q0['g'] += np.abs(np.min(q0['g']))+1e-4
        # Make the max Eq0
        q0['g'] *= Eq0/np.max(q0['g'])
    else:
        q0.set_scales(3/2.)
        q_in = np.loadtxt(IC_file)
        # Center in domain:
        ind_max = np.argmin(np.abs(q_in-np.max(q_in)))
        q_in = np.roll(q_in,-int(ind_max-len(q_in)/2))
        q0['g'][:] = 0.0
        q0['g'][:len(q_in)] = q_in
   
    fh_mode = 'overwrite'
   
else:
    # Restart
    write, last_dt = solver.load_state('restart.h5', -1)

    solver.sim_time = 0.0
    solver.iteration = 1

    # Timestepping and output
#     dt = last_dt
    fh_mode = 'append'

# Integration parameters
solver.stop_sim_time = sim_tmax
solver.stop_wall_time = real_tmax

# #######
# # CFL #
# #######
# CFL = flow_tools.CFL(solver, initial_dt=dt, safety=0.5, cadence=100, threshold=0.05)
# CFL.add_velocity('w',axis=0)
# CFL.add_velocity('wlam',axis=0)
# CFL.add_velocity('a*q*sin(theta)',axis=0)
# # Momentum diffusion
# kcut = np.max(kz)
# CFL.add_frequency(kcut**(2)/Re)
# CFL.add_frequency(drag)
# CFL.add_frequency(kappa)

# logger.info('1/dt restriction from visc = %.3e, drag = %.3e, kappa = %.3e' % (kcut**(2)/Re,drag,kappa))

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
