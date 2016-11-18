#
# AFLOWpi_TB
#
# Utility to construct and operate on TB Hamiltonians from the projections of DFT wfc on the pseudoatomic orbital basis (PAO)
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
from scipy import fftpack as FFT
import numpy as np
import cmath
import sys, time

sys.path.append('./')

def do_spin_orbit_calc(HRaux,natoms,theta,phi,socStrengh):

    nawf = HRaux.shape[0]
    nk1 = HRaux.shape[2]
    nk2 = HRaux.shape[3]
    nk3 = HRaux.shape[4]
    nspin = HRaux.shape[5]

    HR_double= np.zeros((2*nawf,2*nawf,nk1,nk2,nk3,nspin),dtype=complex)
    HR_soc_p = np.zeros((18,18),dtype=complex)  #Hardcoded do s,p,d only (18 orbitals per atom) - Must Change

    # nonmagnetic :  copy H at the upper (lower) left (right) of the double matrix HR_double
    if nspin == 1:
        HR_double[0:nawf,0:nawf,:,:,:,0]   			       	       =  HRaux[0:nawf,0:nawf,:,:,:,0]
        HR_double[nawf:2*nawf,nawf:2*nawf,:,:,:,0] 	      	       =  HRaux[0:nawf,0:nawf,:,:,:,0]
    # magnetic :  copy H_up (H_down) at the upper (lower) left (right) of the double matrix 
    else:
        HR_double[0:nawf,0:nawf,:,:,:,0]   			       	       =  HRaux[0:nawf,0:nawf,:,:,:,0]
        HR_double[nawf:2*nawf,nawf:2*nawf,:,:,:,0] 	       	       =  HRaux[0:nawf,0:nawf,:,:,:,1]

    HR_soc_p =  soc_p(theta,phi)
    #HR_soc_d =  soc_d(theta,phi)

    M=9
    nt=natoms
    for n in range(nt):
        i=n*M
        j=(n+1)*M
        # Up-Up
        HR_double[i:j,i:j,0,0,0,0]                             = HR_double[i:j,i:j,0,0,0,0]                             + socStrengh[n,0]*HR_soc_p[0:9,0:9]
        # Down-Down
        HR_double[(i+nt*M):(j+nt*M),(i+nt*M):(j+nt*M),0,0,0,0] = HR_double[(i+nt*M):(j+nt*M),(i+nt*M):(j+nt*M),0,0,0,0] + socStrengh[n,0]*HR_soc_p[9:18,9:18]
        # Up-Down
        HR_double[i:j,(i+nt*M):(j+nt*M),0,0,0,0]               = HR_double[i:j,(i+nt*M):(j+nt*M),0,0,0,0]               + socStrengh[n,0]*HR_soc_p[0:9,9:18]
        # Down-Up
        HR_double[(i+nt*M):(j+nt*M),i:j,0,0,0,0]               = HR_double[(i+nt*M):(j+nt*M),i:j,0,0,0,0]               + socStrengh[n,0]*HR_soc_p[9:18,0:9]

    return(HR_double)


def soc_p(theta,phi):

    # Hardcoded to s,p,d. This must change latter.
        HR_soc = np.zeros((18,18),dtype=complex) 

        sTheta=cmath.sin(theta)
        cTheta=cmath.cos(theta)

        sPhi=cmath.sin(phi)
        cPhi=cmath.cos(phi)

	#Spin Up - Spin Up  part of the p-satets Hamiltonian
        HR_soc[1,2] = -0.5*np.complex(0.0,sTheta*sPhi)
        HR_soc[1,3] =  0.5*np.complex(0.0,sTheta*cPhi)
        HR_soc[2,3] = -0.5*np.complex(0.0,cTheta)
        HR_soc[2,1]=np.conjugate(HR_soc[1,2])
        HR_soc[3,1]=np.conjugate(HR_soc[1,3])
        HR_soc[3,2]=np.conjugate(HR_soc[2,3])
	#Spin Down - Spin Down  part of the p-satets Hamiltonian
        HR_soc[10:13,10:13] = - HR_soc[1:4,1:4] 
    #Spin Up - Spin Down  part of the p-satets Hamiltonian
        HR_soc[1,11] = -0.5*( np.complex(cPhi,0.0) + np.complex(0.0,cTheta*sPhi))
        HR_soc[1,12] = -0.5*( np.complex(sPhi,0.0) - np.complex(0.0,cTheta*cPhi))
        HR_soc[2,12] =  0.5*np.complex(0.0,sTheta)
        HR_soc[2,10] = -HR_soc[1,11]
        HR_soc[3,10] = -HR_soc[1,12]
        HR_soc[3,11] = -HR_soc[2,12]
	#Spin Down - Spin Up  part of the p-satets Hamiltonian
        HR_soc[11,1]=np.conjugate(HR_soc[1,11])
        HR_soc[12,1]=np.conjugate(HR_soc[1,12])
        HR_soc[10,2]=np.conjugate(HR_soc[2,10])
        HR_soc[12,2]=np.conjugate(HR_soc[2,12])
        HR_soc[10,3]=np.conjugate(HR_soc[3,10])
        HR_soc[11,3]=np.conjugate(HR_soc[3,11])
	return(HR_soc)
