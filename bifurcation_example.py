# cython: linetrace=True
# -*- coding: utf-8 -*-
"""
Example file for simulating a single artery using VaMpy.

.. moduleauthor:: Alexandra K. Diem <alexandra.diem@gmail.com>

"""

from __future__ import division

import sys
import numpy as np
from scipy.interpolate import interp1d

from vampy import *


def inlet(qc, rc, data_dir, f_inlet):
    """
    Function describing the inlet boundary condition. Returns a function.
    """
    Q = np.loadtxt("./%s/%s" % (data_dir, f_inlet), delimiter=',')
    t = [(elem) * qc / rc**3 for elem in Q[:,0]]
    q = [elem / qc for elem in Q[:,1]]
    return interp1d(t, q, kind='linear', bounds_error=False, fill_value=q[0])


def main(param):
    """
    Example main.py for running a VaMpy model of a bifurcation.
    For scaling parameters a and b, see [1]
    [1] M. S. Olufsen et. al. Numerical Simulation and Experimental Validation of Blood Flow in Arteries with Structured-Tree Outflow conditions. Ann. Biomed. Engr., 2000
        
    """
    
    # read config file
    f, a, s = utils.read_config(param) 
    
    # nondimensionalisation parameters
    rc = a['rc'] 
    qc = a['qc']  
    rho = a['rho']
    nu = a['nu']
    Re = qc/(nu*rc)
    
    # assign parameters
    run_id = f['run_id'] # run ID
    f_inlet = f['inlet'] # inlet file
    data_dir = f['data_dir'] # data directory
    T = s['T'] * qc / rc**3 # time of one cycle
    tc = s['tc'] # number of cycles to simulate
    dt = s['dt'] * qc / rc**3 # time step size
    ntr = 50 # number of time steps to be stored
    dx = s['dx'] / rc # spatial step size
    Ru = a['Ru'] / rc # artery radius upstream
    Rd = a['Rd'] / rc # artery radius downstream
    nu = nu*rc/qc # viscosity
    kc = rho*qc**2/rc**4
    k = (a['k1']/kc, a['k2']*rc, a['k3']/kc) # elasticity model parameters (Eh/r)
    #out_args = [a['R1']*rc**4/(qc*rho), a['R2']*rc**4/(qc*rho), 
    #        a['Ct']*rho*qc**2/rc**7] # Windkessel parameters - commented out from original
    
    #Windkessel parameters in iterable format
    #R1 = a['R1']*rc**4/(qc*rho)
    #R2 = a['R2']*rc**4/(qc*rho)
    #Ct = a['Ct']*rho*qc**2/rc**7
    Rt = a['Rt']
    Ct = a['Ct']
    Tau = a['Tau']
    
    out_bc = '3wk'
    p0 = (10 * 1333.22365) * rc**4/(rho*qc**2) # zero transmural pressure
    
    # inlet boundary condition
    q_in = inlet(qc, rc, data_dir, f_inlet)


    # initialise artery network object
    # Set scaling parameters a and b (see [1])
    alpha = 0.9
    beta = 0.6
    #an = ArteryNetwork(Ru, Rd, a['lam'], k, rho, nu, p0, a['depth'], ntr, Re, a=alpha, b=beta)
    an = ArteryNetwork(Ru, Rd, a['lam'], k, rho, nu, p0, a['depth'], ntr, Re)
    an.mesh(dx)
    an.set_time(dt, T, tc)
    an.initial_conditions(0.0)
    [R1j,R2j,Cpj] = an.outflow_conditions(Rt,Ct,Tau) #new equation that solves for R1,R2,Ct given a total resistance
    J = 2**(an.depth -1)
    R1 = [(R1j*1333.22365)*rc**4/(qc*rho)]*J
    R2 = [(R2j*1333.22365)*rc**4/(qc*rho)]*J
    Cp = [(Cpj/1333.22365)*rho*qc**2/rc**7]*J
    # run solver
    #an.solve(q_in, out_bc, out_args) #commented out from originial for iterables
    an.solve(q_in,out_bc,R1,R2,Cp) #added to allow for iterable R1,R2,Ct. out_args became [R1,R2,Ct]
    
    # redimensionalise
    an.redimensionalise(rc, qc)
    
    # save results
    an.dump_results(run_id, f['data_dir'])
        
    
if __name__ == "__main__":
    script, param = sys.argv
    main(param)
