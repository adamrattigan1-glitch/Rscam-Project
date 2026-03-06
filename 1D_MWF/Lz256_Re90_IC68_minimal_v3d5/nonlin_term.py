import numpy as np
from mpi4py import MPI
import time
import pathlib
from dedalus import public as de

import logging
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def nonlin_term_scalar(domain,fields,scalar):
    """
    ns = nonlin_term(domain,[u,v,w],s) 
     or
    ns = nonlin_term(domain,[u,v],s)
    
    Computes and returns the components of:
    -u.grad(s)
    
    """
    # Get bases
    bases = domain.bases
    
    ns = -np.sum([fields[ii]*bases[ii].Differentiate(scalar) for ii in range(len(fields))],axis=0)
    
    ns = ns.evaluate()
    ns.set_scales(1)
    for field in fields:
        field.set_scales(1)
        
    return ns

def nonlin_term(domain,fields):
    """
    [a,b,c] = nonlin_term(domain,[u,v,w,p]) 
     or
    [a,b] = nonlin_term(domain,[u,v,p])
    
    Computes and returns the components of:
    -u.grad(u) - grad(p)
    """
    # Get bases
    bases = domain.bases
    
    if domain.dim==2:
        u,v,p = fields
        a = -u*bases[0].Differentiate(u) -v*bases[1].Differentiate(u) - bases[0].Differentiate(p)
        b = -u*bases[0].Differentiate(v) -v*bases[1].Differentiate(v) - bases[1].Differentiate(p)
        a=a.evaluate()
        b=b.evaluate()

        a.set_scales(1)
        b.set_scales(1)
        u.set_scales(1)
        v.set_scales(1)
        return [a,b]
    elif domain.dim==3:
        u,v,w,p = fields
        a = -u*bases[0].Differentiate(u) -v*bases[1].Differentiate(u)--w*bases[2].Differentiate(u) - bases[0].Differentiate(p)
        b = -u*bases[0].Differentiate(v) -v*bases[1].Differentiate(v)--w*bases[2].Differentiate(v) - bases[1].Differentiate(p)        
        c = -u*bases[0].Differentiate(w) -v*bases[1].Differentiate(w)--w*bases[2].Differentiate(w) - bases[2].Differentiate(p)
        a=a.evaluate()
        b=b.evaluate()
        c=c.evaluate()

        a.set_scales(1)
        b.set_scales(1)
        c.set_scales(1)
        u.set_scales(1)
        v.set_scales(1)
        w.set_scales(1)
        return [a,b,c]
    
    
def pressure_calc(domain,fields):
    """
    p = pressure_calc(domain,[u,v,w]) 
     or
    p = pressure_calc(domain,[u,v])
    
    Computes and returns pressure, by inverting the pressure equation:
    laplacian(p) = -div(u.grad(u))
    
    NOTE: You should only have to use this when initializing fields. Otherwise the pressure is one of the dynamical variables!
    """
    from fractional_laplacian import FractionalLaplacian
    
    # Get bases
    bases = domain.bases
    
    if domain.dim==2:
        u,v = fields
        a = u*bases[0].Differentiate(u) + v*bases[1].Differentiate(u)
        b = u*bases[0].Differentiate(v) + v*bases[1].Differentiate(v)
        p = -FractionalLaplacian(bases[0].Differentiate(a)+bases[1].Differentiate(b),-1)
        p=p.evaluate()
        p.set_scales(1)
    elif domain.dim==3:
        u,v,w = fields
        a = u*bases[0].Differentiate(u) + v*bases[1].Differentiate(u) + w*bases[2].Differentiate(u)
        b = u*bases[0].Differentiate(v) + v*bases[1].Differentiate(v) + w*bases[2].Differentiate(v)
        c = u*bases[0].Differentiate(w) + v*bases[1].Differentiate(w) + w*bases[2].Differentiate(w)
        p = -FractionalLaplacian(bases[0].Differentiate(a)+bases[1].Differentiate(b)+bases[2].Differentiate(c),-1)
        p=p.evaluate()
        p.set_scales(1)
        
    return p
