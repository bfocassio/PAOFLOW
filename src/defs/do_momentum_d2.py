# 
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016,2017 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import numpy as np
import cmath
import os, sys
import scipy.linalg.lapack as lapack

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

from load_balancing import *
from communication import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_momentum(vec,dHksp,d2Hksp,npool):
    # calculate momentum vector

    index = None

    if rank == 0:
        nktot,_,nawf,nawf,nspin = dHksp.shape
        index = {'nawf':nawf,'nktot':nktot,'nspin':nspin}

    index = comm.bcast(index,root=0)

    nktot = index['nktot']
    nawf = index['nawf']
    nspin = index['nspin']

    if rank == 0:
        pksp = np.zeros((nktot,3,nawf,nawf,nspin),dtype=complex)
        tksp = np.zeros((nktot,3,3,nawf,nawf,nspin),dtype=complex)
    else:
        dHksp = None
        pksp = None
        tksp = None

    for pool in xrange(npool):
        if nktot%npool != 0: sys.exit('npool not compatible with MP mesh - do_momentum')
        nkpool = nktot/npool

        if rank == 0:
            dHksp_split = np.array_split(dHksp,npool,axis=0)[pool]
            d2Hksp_split = np.array_split(d2Hksp,npool,axis=0)[pool]
            pks_split = np.array_split(pksp,npool,axis=0)[pool]
            tks_split = np.array_split(tksp,npool,axis=0)[pool]
            vec_split = np.array_split(vec,npool,axis=0)[pool]
        else:
            dHksp_split = None
            d2Hksp_split = None
            pks_split = None
            tks_split = None
            vec_split = None

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        nsize = end_ik-ini_ik

        comm.Barrier()
        dHkaux = scatter_array(dHksp_split)
        d2Hkaux = scatter_array(d2Hksp_split)
        pksaux = scatter_array(pks_split)
        tksaux = scatter_array(tks_split)
        vecaux = scatter_array(vec_split)

        for ik in xrange(nsize):
            for ispin in xrange(nspin):
                for l in xrange(3):
                    pksaux[ik,l,:,:,ispin] = np.conj(vecaux[ik,:,:,ispin].T).dot \
                                (dHkaux[ik,l,:,:,ispin]).dot(vecaux[ik,:,:,ispin])
                    for lp in xrange(3):
                        tksaux[ik,l,lp,:,:,ispin] = np.conj(vecaux[ik,:,:,ispin].T).dot \
                                    (d2Hkaux[ik,l,lp,:,:,ispin]).dot(vecaux[ik,:,:,ispin])

        comm.Barrier()
        gather_array(pks_split, pksaux)
        gather_array(tks_split, tksaux)

        if rank == 0:
            pksp[pool*nkpool:(pool+1)*nkpool,:,:,:,:] = pks_split[:,:,:,:,:,]
            tksp[pool*nkpool:(pool+1)*nkpool,:,:,:,:,:] = tks_split[:,:,:,:,:,:,]

    return(pksp,tksp)
