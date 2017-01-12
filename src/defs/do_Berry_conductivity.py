#
# PAOpy
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016 ERMES group (http://ermes.unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
#
# References:
# Luis A. Agapito, Andrea Ferretti, Arrigo Calzolari, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Effective and accurate representation of extended Bloch states on finite Hilbert spaces, Phys. Rev. B 88, 165127 (2013).
#
# Luis A. Agapito, Sohrab Ismail-Beigi, Stefano Curtarolo, Marco Fornari and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonian Matrices from Ab-Initio Calculations: Minimal Basis Sets, Phys. Rev. B 93, 035104 (2016).
#
# Luis A. Agapito, Marco Fornari, Davide Ceresoli, Andrea Ferretti, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonians for 2D and Layered Materials, Phys. Rev. B 93, 125137 (2016).
#
# Pino D'Amico, Luis Agapito, Alessandra Catellani, Alice Ruini, Stefano Curtarolo, Marco Fornari, Marco Buongiorno Nardelli, 
# and Arrigo Calzolari, Accurate ab initio tight-binding Hamiltonians: Effective tools for electronic transport and 
# optical spectroscopy from first principles, Phys. Rev. B 94 165166 (2016).
# 

import numpy as np
import cmath
from math import cosh
import sys, time
import scipy.integrate

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from load_balancing import load_balancing

from constants import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_Berry_conductivity(E_k,pksp,temp,ispin,npool,ipol,jpol):
    # Compute the optical conductivity tensor sigma_xy(ene)

    emin = 0.0 # To be read in input
    emax = 10.0
    de = (emax-emin)/500
    ene = np.arange(emin,emax,de,dtype=float)

    index = None

    if rank == 0:
        nktot,_,nawf,_,nspin = pksp.shape
        index = {'nktot':nktot,'nawf':nawf,'nspin':nspin}

    index = comm.bcast(index,root=0)

    nktot = index['nktot']
    nawf = index['nawf']
    nspin = index['nspin']

    sigxy = np.zeros((ene.size),dtype=complex)
    sigxy_sum = np.zeros((ene.size),dtype=complex)

    for pool in xrange(npool):

        if nktot%npool != 0: sys.exit('npool not compatible with MP mesh')
        nkpool = nktot/npool

        if rank == 0:
            pksp_long = np.array_split(pksp,npool,axis=0)[pool]
            E_k_long= np.array_split(E_k,npool,axis=0)[pool]
        else:
            pksp_long = None
            E_k_long = None

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        nsize = end_ik-ini_ik

        pkspaux = np.zeros((nsize,3,nawf,nawf,nspin),dtype=complex)
        E_kaux = np.zeros((nsize,nawf,nspin),dtype=float)
        sigxy_aux = np.zeros((ene.size),dtype=complex)

        comm.Barrier()
        comm.Scatter(pksp_long,pkspaux,root=0)
        comm.Scatter(E_k_long,E_kaux,root=0)

        sigxy_aux[:] = sigma_loop(ini_ik,end_ik,ene,E_kaux,pkspaux,nawf,temp,ispin,ipol,jpol)

        comm.Allreduce(sigxy_aux,sigxy_sum,op=MPI.SUM)
        sigxy += sigxy_sum

    sigxy /= float(nktot)

    return(ene,sigxy)

def sigma_loop(ini_ik,end_ik,ene,E_k,pksp,nawf,temp,ispin,ipol,jpol):

    sigxy = np.zeros((ene.size),dtype=complex)
    func = np.zeros((end_ik-ini_ik,ene.size),dtype=complex)
    delta = 0.05

    # Collapsing the sum over k points
    for n in xrange(nawf):
        fn = 1.0/(np.exp(E_k[:,n,ispin]/temp)+1)
        for m in xrange(nawf):
            fm = 1.0/(np.exp(E_k[:,m,ispin]/temp)+1)
            func[:,:] = ((E_k[:,n,ispin]-E_k[:,m,ispin])**2*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T - (ene+1.0j*delta)**2
            sigxy[:] += np.sum(((1.0/func * \
                        ((fn - fm)*np.ones((end_ik-ini_ik,ene.size),dtype=float).T).T).T* \
                        np.imag(pksp[:,jpol,n,m,0]*pksp[:,ipol,m,n,0]-pksp[:,ipol,n,m,0]*pksp[:,jpol,m,n,0])
                        ),axis=1)

    return(sigxy)

