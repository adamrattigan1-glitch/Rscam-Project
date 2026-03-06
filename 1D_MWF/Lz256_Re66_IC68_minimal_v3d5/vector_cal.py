import numpy as np
from mpi4py import MPI
import time
import pathlib
from dedalus import public as de

import logging
logger = logging.getLogger(__name__)

def curl(domain,fields):
    """
    Options:
    [Fx,Fy] = curl(domain,[psi])
    Fz = curl(domain,[u,v])
    [Fx,Fy,Fz] = curl(domain,[u,v,w])

    input: domain, fields (list of dedalus fields, like [u,v,w], [u,v], or [psi])
    output: operators, [curl(u,v,w)_0,curl(u,v,w)_1,curl(u,v,w)_2], [curl(u,v)], or [curl(psi z)_0,curl(psi)_1] (respectively), the curl of the vectors you input.
    
    NOTE: you can feed it 1, 2 or 3 component data. It will interpret accordingly. 
    However, make sure to maintain the right-handedness (i.e. put the entries in order
    corresponding to the axes, i.e. u,v,w for x,y,z)
    """
    # Get bases
    bases = domain.bases

    if len(fields)==1:
        dCdx = bases[0].Differentiate(fields[0])
        dCdy = bases[1].Differentiate(fields[0])
        Fx = dCdy
        Fy = -dCdx
        return [Fx,Fy]
    
    if len(fields)==2:
        dBdx = bases[0].Differentiate(fields[1])
        dAdy = bases[1].Differentiate(fields[0])
        Fz = dBdx-dAdy
        return Fz
    
    elif len(fields)==3:
        dAdy = bases[1].Differentiate(fields[0])
        dAdz = bases[2].Differentiate(fields[0])

        dBdx = bases[0].Differentiate(fields[1])
        dBdz = bases[2].Differentiate(fields[1])

        dCdx = bases[0].Differentiate(fields[2])
        dCdy = bases[1].Differentiate(fields[2])
        
        Fx = dCdy - dBdz
        Fy = dAdz - dCdx
        Fz = dBdx - dAdy
        return [Fx,Fy,Fz]
        
        
def div(domain,fields):
    """
    div = div(domain,[u,v,w])
    
    input: domain, fields (list of dedalus fields, like [u,v,w], or [u,v])
    output: field, the divergence of the vectors you input.
    
    It will take the field you feed it and calculate dx(u)+dy(v) + ...
    Even if you're in a 3D domain, feeding it two vectors will result in dx(u)+dy(v)
    """
    # Get bases
    bases = domain.bases

    div = np.sum([bases[ii].Differentiate(fields[ii]) for ii in range(len(fields))],axis=0)
    
    return div
        
def lap(domain,u):
    """
    lap = lap(domain,u)
    input: domain, u (a field)
    output: field, laplacian of input field
    """
    # Get bases
    bases = domain.bases

    for ii,base in enumerate(bases):
        if ii==0:
            lap = base.Differentiate(base.Differentiate(u))
        else:
            lap += base.Differentiate(base.Differentiate(u))

    return lap
