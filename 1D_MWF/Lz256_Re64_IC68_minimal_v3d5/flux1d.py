import numpy as np
from mpi4py import MPI
import time
import pathlib
from dedalus import public as de

import logging
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def flux1d_classic(domain,fields,nonlins,flux=True):
    """
    [modn,Pi] = flux1d_classic(domain,[u,v,w],[a,b,c]) or
    [modn,Pi] = flux1d_classic(domain,[u,v],[a,b])

    input: domain, field1, field2 
             - field1 is a vector and field2 is the nonlinear term
    output: [modn,Pi], spherically averaged energy transfer from shell to shell. modn is the mode number magnitude (not the wavenumber)

    Options:
    - flux(=True by default): if False, then outputs just the transfers, otherwise it integrates and gives the flux Pi.

    Notes: 
     - You can feed it 2D or 3D data. It will always average over angles to make 
    an isotropic spectrum
     - This version does the 'classic' calculation of the flux, not taking into account the fact that for large |k| less modes are represented.
    """
    # Construct the modes
    ns = []
    for i in range(domain.dim):
        k = domain.elements(i)
        L = np.sum(domain.bases[i].grid_spacing())
        n = np.round(k*(L/(2*np.pi))).astype(np.int)
        ns.append(n)
        
    n2 = np.sqrt(np.sum([n**2 for n in ns],axis=0))
     
    # Find maximum mode number modulus:
    n2max = np.round(np.amax(n2)).astype(np.int)
    n2max_all = 0.0
    n2max_all = comm.allreduce(n2max,op=MPI.MAX)
              
    # square them and add them up to get energy density        
    KEs = np.sum([np.real(fields[ii]['c']*np.conjugate(nonlins[ii]['c'])) for ii in range(len(fields))],axis=0)
    KEs *= (2 - (ns[0] == 0))  # Double-count positive kx because of R2C transform

    # Construct arrays for final output
    modn = np.arange(1,n2max_all+1)
    En = np.zeros(n2max_all)   
    
    for k in modn:
        En[modn==k] = np.sum(KEs[(np.round(n2).astype(np.int))==k])
        
    # Do zeroth by itself:
    En[modn==1] += np.sum(KEs[(np.round(n2).astype(np.int))==0])
    
    Pi = comm.allreduce(En,op=MPI.SUM)
    
    if flux:
        Pi = np.cumsum(Pi)

    return [modn,Pi]


def flux1d(domain,fields,nonlins,flux=True):
    """
    [modn,Pi] = flux1d_classic(domain,[u,v,w],[a,b,c]) or
    [modn,Pi] = flux1d_classic(domain,[u,v],[a,b])

    input: domain, field1, field2 
             - field1 is a vector and field2 is the nonlinear term
    output: [modn,Pi], spherically averaged energy transfer from shell to shell
    
    Options:
    - flux(=True by default): if False, then outputs just the transfers, otherwise it integrates and gives the flux Pi.
    
    NOTES/WARNINGS: 
     - You can feed it 2D or 3D data. It will always average over angles to make 
    an isotropic spectrum
     - This version compensates the fact that some modes are less represented in a spherical shell with a cartesian lattice of wavenumbers. 
     - This version is also 10x faster than the flux1d_classic, so use this one if possible!
    """
    # Construct the modes
    ns = []
    for i in range(domain.dim):
        k = domain.elements(i)
        L = np.sum(domain.bases[i].grid_spacing())
        n = np.round(k*(L/(2*np.pi))).astype(np.int)
        ns.append(n)
        dn = 1
        
    n2 = np.sqrt(np.sum([n**2 for n in ns],axis=0)) # not n^2, just |n|
    
    # Find maximum mode number modulus:
    n2max = np.round(np.amax(n2)).astype(np.int)
    n2max_all = comm.allreduce(n2max,op=MPI.MAX)
    
    # Take histogram of modes lying in each spherical shell
    # Make the first interval from 0 to 1.5, then the rest add one to it.
    bins = np.concatenate(([0.0],np.arange(1.5, n2max_all+1.5, 1)))
        
    hist_samples, _ = np.histogram(n2, bins=bins)
    hist_samples = comm.allreduce(hist_samples,op=MPI.SUM)
              
    # square feilds and add them up to get energy density        
    KEs = np.sum([np.real(fields[ii]['c']*np.conjugate(nonlins[ii]['c'])) for ii in range(len(fields))],axis=0)
    if domain.dim==2:
        KEs_1d = KEs * np.pi * n2
    elif domain.dim==3:
        KEs_1d = KEs * 4 * np.pi * n2**2
    else:
        print("ERROR: can only compute power spectrum from 2 or 3 dimensional data.")
    KEs_1d *= (2 - (ns[0] == 0))  # Double-count positive kx because of R2C transform

    # Plot histogram
    pow_samples, _ = np.histogram(n2, bins=bins, weights=KEs_1d)
    pow_samples = comm.allreduce(pow_samples,op=MPI.SUM)
    
    Pi = pow_samples / hist_samples
    
    if flux:
        Pi = np.cumsum(Pi) # THIS ASSUMES dn = 1, need to adjust accordingly
        
    logger.error('WARNING: You are using the version of the flux calculation which does not seem to conserve energy. Try using "flux1d_classic", which is about 8 times slower, but conserves the energy.')
    
    return [np.arange(1,len(Pi)+1),Pi]


def flux1d_scalar_classic(domain,fields,nonlins,flux=True):
    """
    [modn,Pi] = flux1d_scalar_classic(domain,s,ns) or
    [modn,Pi] = flux1d_scalar_classic(domain,s,ns)

    input: domain, field1, field2 
             - field1 is a scalar field and field2 is the nonlinear term of the scalar: u.grad(s).
             - You can calculate field2 using 'nonlin_term_scalar' function in the nonlin_term file.
    output: [modn,Pi], spherically averaged energy transfer from shell to shell. modn is the mode number magnitude (not the wavenumber)

    Options:
    - flux(=True by default): if False, then outputs just the transfers, otherwise it integrates and gives the flux Pi.

    Notes: 
     - You can feed it 2D or 3D data. It will always average over angles to make 
    an isotropic spectrum
     - This version does the 'classic' calculation of the flux, not taking into account the fact that for large |k| less modes are represented.
    """
    # Construct the modes
    ns = []
    for i in range(domain.dim):
        k = domain.elements(i)
        L = np.sum(domain.bases[i].grid_spacing())
        n = np.round(k*(L/(2*np.pi))).astype(np.int)
        ns.append(n)
        
    n2 = np.sqrt(np.sum([n**2 for n in ns],axis=0))
     
    # Find maximum mode number modulus:
    n2max = np.round(np.amax(n2)).astype(np.int)
    n2max_all = 0.0
    n2max_all = comm.allreduce(n2max,op=MPI.MAX)
              
    # square them and add them up to get energy density        
    KEs = np.real(fields['c']*np.conjugate(nonlins['c']))
    KEs *= (2 - (ns[0] == 0))  # Double-count positive kx because of R2C transform

    # Construct arrays for final output
    modn = np.arange(1,n2max_all+1)
    En = np.zeros(n2max_all)   

    # Do zeroth first:
    En[modn==1] = np.sum(KEs[(np.round(n2).astype(np.int))==0])
    
    for k in modn:
        En[modn==k] = np.sum(KEs[(np.round(n2).astype(np.int))==k])
    
    Pi = comm.allreduce(En,op=MPI.SUM)
    
    if flux:
        Pi = np.cumsum(Pi)

    return [modn,Pi]