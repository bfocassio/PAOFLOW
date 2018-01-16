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
import sys
from numpy import linalg as LAN
from scipy import linalg as LA

def do_non_ortho(Hks,Sks):
    # Take care of non-orthogonality, if needed
    # Hks from projwfc is orthogonal. If non-orthogonality is required, we have to apply a basis change to Hks as
    # Hks -> Sks^(1/2)*Hks*Sks^(1/2)

    nawf,_,nkpnts,nspin = Hks.shape
    S2k  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
    for ik in range(nkpnts):
        S2k[:,:,ik] = LA.sqrtm(Sks[:,:,ik])

    Hks_no = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
    for ispin in range(nspin):
        for ik in range(nkpnts):
            Hks_no[:,:,ik,ispin] = np.dot(S2k[:,:,ik],Hks[:,:,ik,ispin]).dot(S2k[:,:,ik])

    return(Hks_no)
