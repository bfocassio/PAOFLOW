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

def do_momentum(vec,dHksp,npool):
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
    else:
        dHksp = None
        pksp = None

    for pool in xrange(npool):
        ini_ip, end_ip = load_balancing(npool,pool,nktot)
        nkpool = end_ip - ini_ip

        if rank == 0:
            dHksp_split = dHksp[ini_ip:end_ip]
            pks_split = pksp[ini_ip:end_ip]
            vec_split = vec[ini_ip:end_ip]
        else:
            dHksp_split = None
            pks_split = None
            vec_split = None

        # Load balancing
        ini_ik, end_ik = load_balancing(size,rank,nkpool)
        nsize = end_ik-ini_ik

        dHkaux = scatter_array(dHksp_split)
        pksaux = scatter_array(pks_split)
        vecaux = scatter_array(vec_split)

        for ik in xrange(nsize):
            for ispin in xrange(nspin):
                for l in xrange(3):
                    pksaux[ik,l,:,:,ispin] = np.conj(vecaux[ik,:,:,ispin].T).dot \
                                (dHkaux[ik,l,:,:,ispin]).dot(vecaux[ik,:,:,ispin])

        gather_array(pks_split, pksaux)

        if rank == 0:
            pksp[ini_ip:end_ip,:,:,:,:] = pks_split[:,:,:,:,:]

    return(pksp)
