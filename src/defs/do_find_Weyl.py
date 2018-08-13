#
# PAOpy
#
# Utility to construct and operate on Hamliltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
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
# Pino D'Amico, Luis gapito, Alessandra Catellani, Alice Ruini, Stefano Curtarolo, Marco Fornari, Marco Buongiorno Nardelli, 
# and Arrigo Calzolari, Accurate ab initio tight-binding Hamiltonians: Effective tools for electronic transport and 
# optical spectroscopy from first principles, Phys. Rev. B 94 165166 (2016).
# 




from scipy import fftpack as FFT
import numpy as np
import cmath
import sys
import scipy
try:
    import pyfftw
except:
    from scipy import fftpack as FFT
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from kpnts_interpolation_mesh import *
from do_non_ortho import *
import do_momentum
#import do_gradient
import os
#from load_balancing import *
from get_K_grid_fft import *
from constants import *
import time
import scipy.optimize as OP
from numpy import linalg as LAN
from load_balancing import *
from do_double_grid import *
#import do_bandwarping_calc

from clebsch_gordan import *
from get_R_grid_fft import *
#import nxtval
from communication import gather_full
#from Gatherv_Scatterv_wrappers import Gatherv_wrap


# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

np.set_printoptions(precision=6, threshold=100, edgeitems=50, linewidth=350, suppress=True)
load=False

from numpy import linalg as LAN


en_range=0.50
#en_range=5.50
#first_thresh=1.e-4
first_thresh=0.01

def gen_eigs(HRaux,kq,Rfft,band,b_vectors):
    # Load balancing

##    kq=kq.dot(b_vectors.T)
    kq = np.dot(kq,b_vectors)
    read_S=False
    nawf,nawf,_,nspin = HRaux.shape
    E_kp = np.zeros((1,nawf,nspin),dtype=np.float64)
    kq=np.array([kq,],dtype=complex).T
    nkpi=1

    Hks_int  = np.zeros((nawf,nawf,1,nspin),dtype=complex,order='C') # final data arrays
    Hks_int = band_loop_H(0,1,nspin,nawf,nkpi,HRaux,kq,Rfft)

    v_kp  = np.zeros((1,nawf,nawf,nspin),dtype=np.complex)
    for ispin in range(nspin):
        E_kp[:,:,ispin],v_kp[:,:,:,ispin] =  LAN.eigh(Hks_int[:,:,0,ispin],UPLO='U')
    #print(E_kp[0,band,0])
    return E_kp.real,v_kp



def get_Sj(spol,nawf,spin_orbit,nl,sh):

    # Compute spin current matrix elements
    # Pauli matrices (x,y,z)
    sP=0.5*np.array([[[0.0,1.0],[1.0,0.0]],[[0.0,-1.0j],[1.0j,0.0]],[[1.0,0.0],[0.0,-1.0]]])
    if spin_orbit:
        # Spin operator matrix  in the basis of |l,m,s,s_z> (TB SO)
        Sj = np.zeros((nawf,nawf),dtype=complex)
        for i in range(nawf/2):
            Sj[i,i] = sP[spol][0,0]
            Sj[i,i+1] = sP[spol][0,1]
        for i in range(nawf/2,nawf):
            Sj[i,i-1] = sP[spol][1,0]
            Sj[i,i] = sP[spol][1,1]
    else:
        # Spin operator matrix  in the basis of |j,m_j,l,s> (full SO)
        Sj = clebsch_gordan(nawf,sh,nl,spol)


    return Sj


def get_spin(Sj,HRs,Rfft,b_vectors,band,kq,):
    # Load balancing
    ini_ik, end_ik = 0,1
    nkpi=1
    nspin=1




    _,v_kp = gen_eigs(HRs,kq,Rfft,band,b_vectors)




    #ipol


    sp = np.zeros((3),dtype=float)

    for l in range(3):
        sp[l] = np.real(np.conj(v_kp[0,:,:,0].T).dot(Sj[l,:,:]).dot(v_kp[0,:,:,0]))[band,band]

    return sp




def find_egap(HRaux,kq,Rfft,band,b_vectors,ef_index,ispin):
#    kq=np.array([0.369234,-0.002,0.004])
#    kq=np.array([1.0/3.0,1.0/3.0,0.5])
#    kq = np.array([0,0,0])
#    kq = np.array([-0.5,0.5,0.5])
#    kq=kq.dot(b_vectors.T)
#    kq=np.dot(kq,b_vectors)
    E_kp,_= gen_eigs(HRaux,kq,Rfft,band,b_vectors)
#  print (E_kp.shape)
    egapp = E_kp[0,ef_index,ispin]-E_kp[0,ef_index-1,ispin]
#    print (E_kp[0,:,:])
    if egapp<0.00001:
        print ('kq_frac:{},kq_cart:{},Egap:{}eV,between band#{}at{}eV and band#{}at{}eV'.format(kq,np.dot(kq,b_vectors),egapp,ef_index+1,E_kp[0,ef_index,ispin],ef_index,E_kp[0,ef_index-1,ispin]))
    return egapp




def do_search_grid(nk1,nk2,nk3,snk1_range=[-0.5,0.5],snk2_range=[-0.5,0.5],snk3_range=[-0.5,0.5],endpoint=False):

    nk1_arr   = np.linspace(snk1_range[0],snk1_range[1],num=nk1,   endpoint=endpoint)
    nk2_arr   = np.linspace(snk2_range[0],snk2_range[1],num=nk2,   endpoint=endpoint)
    nk3_arr   = np.linspace(snk3_range[0],snk3_range[1],num=nk3,   endpoint=endpoint)



    nk_str = np.zeros((nk1*nk2*nk3,3),order='C')
    nk_str  = np.array(np.meshgrid(nk1_arr,nk2_arr,nk3_arr,indexing='ij')).T.reshape(-1,3)

    return nk_str



def find_min(ef_index,HRaux,SRaux,read_S,alat,velkp1,nk1,nk2,nk3,bnd,nspin,a_vectors,b_vectors,v_k,snk1_range=[-0.5,0.5],snk2_range=[-0.5,0.5],snk3_range=[-0.5,0.5],npool=1,shift=0.0,nl=None,sh=None):
    band = bnd
    np.set_printoptions(precision=6, threshold=100, edgeitems=50, linewidth=350, suppress=True)
    comm.Barrier()


    velkp = None

    kq_temp,_,_,_ = get_K_grid_fft(nk1,nk2,nk3,np.identity(3))
    kq_temp=kq_temp.T



    nk1+=1
    nk2+=1
    nk3+=1

    snk1=snk2=snk3=10

    search_grid = do_search_grid(snk1,snk2,snk3,
                                 snk1_range=[-0.51,0.51],
                                 snk2_range=[-0.51,0.51],
                                 snk3_range=[-0.51,0.51],
                                 endpoint=False)

    #offset the search grid so it doesn't find the same 
    #extrema at corners/edges
    ini_ik, end_ik = load_balancing(size,rank,search_grid.shape[0])


    #do the bounds for each search subsection of FBZ
    bounds_K  = np.zeros((search_grid.shape[0],3,2))
    guess_K   = np.zeros((search_grid.shape[0],3))
    #full grid

    #get the search grid off possible HSP

    #search in boxes
    bounds_K[:,:,0] = search_grid
    bounds_K[:,:,1] = search_grid+1.0*np.array([1.02/(snk1),1.02/(snk2),1.02/(snk3)])

    #initial guess is in the middle of each box
    guess_K = search_grid+np.array([0.25/snk1,0.25/snk2,0.25/snk3])


    #partial grid

    #swap bounds axes to put in the root finder
    bounds_K=np.swapaxes(bounds_K,1,2)

    #add two extra columns for info on which band and what spin for the extrema
    #so we don't lose informatioen we reshape to reduce the duplicate entries

    all_extrema_shape = [snk1*snk2*snk3,bnd,nspin,6]
    all_extrema_shape[0] =snk1*snk2*snk3


    vk_mesh = None
    velkp   = None
    velkp1  = None
    velkp1p = None
    velkp1=None
    comm.Barrier()

    sst=0
#    counter = nxtval.Counter(comm,init=sst)



    if rank==0:
        nawf,nawf,nr1,nr2,nr3,nspin = HRaux.shape
        HRaux = HRaux.reshape((nawf,nawf,nr1*nr2*nr3,nspin))
    else:
        nspin=nawf=nr1=nr2=nr3=None

    nspin = comm.bcast(nspin)
    nawf  = comm.bcast(nawf)
    nr1   = comm.bcast(nr1)
    nr2   = comm.bcast(nr2)
    nr3   = comm.bcast(nr3)


    if rank!=0:
        HRaux=np.zeros((nawf,nawf,nr1*nr2*nr3,nspin),dtype=complex,order='C')

    comm.Barrier()
    comm.Bcast(HRaux)
    comm.Barrier()

    weyl=True


    R,_,_,_,_ = get_R_grid_fft(nr1,nr2,nr3,a_vectors)
    Rfft=R

    ll=30
    if (bnd-ll)>=0:
        st=bnd-ll
    else:
        st=0
#    sg = np.array(np.meshgrid(xrange(guess_K.shape[0]),xrange(bnd),xrange(nspin),indexing='ij')).T.reshape(-1,3)
#    sg = np.array(np.meshgrid(xrange(guess_K.shape[0]),xrange(band),xrange(nspin),indexing='ij')).T.reshape(-1,3)
    sg = np.array(np.meshgrid(range(guess_K.shape[0]),list(range(1)),range(nspin),indexing='ij')).T.reshape(-1,3)
    if rank ==0:print (sg)
    comm.Barrier()

    fold_coord = lambda x: ((x+0.5)%1.0-0.5)

    extrema=[]
    timer_avg = 0.0

    if rank==0:
        print("starting find crit at grid %s"%sst)
    fp=0
    sp=0
    ep=0

    candidates = 1
    c = -1
    for i in range(sg.shape[0]//size+1):

        c+=1
#        c = next(counter)

        if c>=sg.shape[0]:
            continue

        i     = sg[c][0]
        b     = sg[c][1]
        ispin = sg[c][2]

        if weyl:
            lam_XiP  = lambda K: find_egap(HRaux,K,Rfft,band,b_vectors,ef_index,ispin)
#loop over the search grid 
##        try:            

#            eig,_ = gen_eigs(HRaux,guess_K[i],Rfft,b,b_vectors)

#            if np.abs(eig[0,b,0]-shift)<0.15 or np.abs(eig[0,b,0])>en_range:
#                continue
#            else: 
#                ep+=1

            startt = time.time()
            #fist pass
            solx = OP.least_squares(lam_XiP,guess_K[i],bounds=bounds_K[i],jac='2-point',max_nfev=30000)
#                                    method='trf',jac='2-point',ftol=9.9e-3,
#                                    max_nfev=300)

            print (solx.fun)
            nfvf = solx.nfev
            #second pass
            if np.all(np.abs(solx.fun)<0.01):
#                print ('starting second pass')
                print ('Candidate No.{} found'.format(candidates))
                print (solx.x)
                print (solx.fun)
                candidates += 1
#                solx = OP.least_squares(lam_XiP,solx.x,bounds=bounds_K[i],jac='3-point',max_nfev=30000)
#ftol=5.e-16,gtol=1.e-14,
#                                        method='trf',jac='3-point',xtol=1.e-12,
#                                        max_nfev=300)
            else:
                fp+=1
                continue

        if np.all(np.abs(solx.fun)<=first_thresh):

            solxx = solx.x

            eigs,_  = gen_eigs(HRaux,solxx,Rfft,b,b_vectors)
            solxx = solxx.dot(b_vectors.T)
            en = find_egap(HRaux,solxx,Rfft,band,b_vectors,ef_index,ispin)
            print("flag % 6d / %d  % 4d  "%(c+1,sg.shape[0],nfvf),"% 5.4f % 5.4f % 5.4f % 6.6f"%\
                       (solxx[0],solxx[1],solxx[2],en))

            extrema.append([solxx[0],solxx[1],solxx[2],en])

    if len(extrema)==0:
        extrema=np.zeros((2,6),dtype=float,order="C")
    else:
        extrema = np.asarray(extrema,dtype=float,order="C")


    comm.Barrier()
    all_extrema = Gatherv_wrap(extrema)

    if rank==0:
        zero_mask   = np.sum(all_extrema,axis=1)!=0.0
        all_extrema = all_extrema[zero_mask]
        if weyl:
            np.savetxt('weyl.dat',all_extrema)
