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



import z2pack
import tbmodels
import scipy.optimize as so
#import matplotlib.pyplot as plt
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


def band_loop_H(ini_ik,end_ik,nspin,nawf,nkpi,HRaux,kq,R):

   nawf,nawf,_,nspin = HRaux.shape
   kdot = np.zeros((1,R.shape[0]),dtype=complex,order="C")
   kdot = np.tensordot(R,2.0j*np.pi*kq[:,ini_ik:end_ik],axes=([1],[0]))
   np.exp(kdot,kdot)

   auxh = np.zeros((nawf,nawf,1,nspin),dtype=complex,order="C")
   for ispin in range(nspin):
       auxh[:,:,ini_ik:end_ik,ispin]=np.tensordot(HRaux[:,:,:,ispin],kdot,axes=([2],[0]))
   kdot  = None
   return(auxh)



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
#    if egapp<0.00001:
#        print ('kq_frac:{},kq_cart:{},Egap:{}eV,between band#{}at{}eV and band#{}at{}eV'.format(kq,np.dot(kq,b_vectors),egapp,ef_index+1,E_kp[0,ef_index,ispin],ef_index,E_kp[0,ef_index-1,ispin]))
    return egapp




def do_search_grid(nk1,nk2,nk3,snk1_range=[-0.5,0.5],snk2_range=[-0.5,0.5],snk3_range=[-0.5,0.5],endpoint=False):

    nk1_arr   = np.linspace(snk1_range[0],snk1_range[1],num=nk1,   endpoint=endpoint)
    nk2_arr   = np.linspace(snk2_range[0],snk2_range[1],num=nk2,   endpoint=endpoint)
    nk3_arr   = np.linspace(snk3_range[0],snk3_range[1],num=nk3,   endpoint=endpoint)



    nk_str = np.zeros((nk1*nk2*nk3,3),order='C')
    nk_str  = np.array(np.meshgrid(nk1_arr,nk2_arr,nk3_arr,indexing='ij')).T.reshape(-1,3)

    return nk_str





def loop_min(ef_index,HRaux,SRaux,read_S,alat,velkp1,nk1,nk2,nk3,bnd,nspin,a_vectors,b_vectors,v_k,snk1_range=[-0.5,0.5],snk2_range=[-0.5,0.5],snk3_range=[-0.5,0.5],npool=1,shift=0.0,nl=None,sh=None):
    ini=[[0.25,0.25,0.25],[0.25,0.25,0.75],[0.25,0.75,0.25],[0.25,0.75,0.75],[0.75,0.25,0.25],[0.75,0.25,0.75],[0.75,0.75,0.25],[0.75,0.75,0.75]]
    CANDIDATES = {}
#    CANDIDATES = {'[-0.007865 -0.33889  -0.033505]': [6.6127342213118112e-08], '[0.180072 0.179205 0.005416]': [1.523763586017135e-06], '[-0.507005 -0.508404 0.094956]': [7.777151428328688e-06], '[-0.491938 -0.016849 0.094221]': [3.189758233310136e-06], '[0.345998 -0.338489 0.033075]': [2.2020509471282201e-06], '[-0.008887 0.348185 0.034663]': [5.9531129448231468e-07], '[ 0.492577  0.492016 -0.094952]': [4.4537277532619157e-08], '[-0.492995 0.508404 -0.094956]': [7.5562735711992568e-06], '[-0.492577 0.507984 0.094952]': [4.4379632659213852e-06], '[-0.006277 0.34548 -0.033989]': [4.1678444084869426e-06], '[-0.34551 0.006347 -0.034031]': [2.9474345103197575e-06], '[ 0.005803  0.339145 -0.033455]': [2.3650376379524829e-08], '[ 0.491938  0.016849 -0.094221]': [7.9981158307429467e-08], '[ 0.348214 -0.008945 -0.034718]': [6.2017560392702009e-08], '[-0.348214 0.008945 0.034718]': [6.6011608114424636e-07], '[-0.345998  0.338489 -0.033075]': [6.5991954220634419e-08], '[ 0.344701 -0.338921 -0.033313]': [3.5319852778603611e-08], '[0.175098 -0.358254 0.005226]': [6.2559472217837975e-07], '[-0.016879 -0.485697 0.094325]': [4.8522886401530796e-06], '[0.007865 0.33889 0.033505]': [5.9198951841976655e-07], '[ 0.492577 -0.507984 -0.094952]': [4.8026373530851707e-08], '[0.507423 -0.492016 0.094952]': [4.3749342975296646e-06], '[0.507005 -0.491596 -0.094956]': [7.4392469254919158e-06], '[ 0.179294  0.179983 -0.005416]': [6.7186063379409688e-08], '[-0.507423  0.492016 -0.094952]': [6.0839751347963045e-08], '[-0.33888 -0.007829 0.033472]': [1.7667997588843853e-06], '[ 0.008887 -0.348185 -0.034663]': [1.3431241871475486e-07], '[-0.486367 -0.01688 -0.094333]': [3.4468396935549706e-06], '[0.175719 -0.358357 -0.005243]': [2.9719146784207284e-06], '[-0.35836 0.175699 0.005238]': [1.6496662409853924e-06], '[ 0.508062 -0.016849  0.094221]': [6.0724664449618437e-08], '[-0.338955 0.344642 0.033263]': [1.2423807027872602e-06], '[-0.005803 -0.339145 0.033455]': [1.1170258334275429e-06], '[-0.492577 -0.492016 0.094952]': [4.3190014239746777e-06], '[-0.016844 -0.491229 -0.094208]': [1.8106358440350689e-08], '[-0.175719  0.358357  0.005243]': [4.3814737823999472e-08], '[ 0.33888   0.007829 -0.033472]': [2.3856515554698809e-08], '[0.339136 0.005772 0.033433]': [1.7278282039429049e-06], '[-0.492995 -0.491596 -0.094956]': [6.6824492619965703e-08], '[ 0.507005  0.508404 -0.094956]': [6.5442281552141601e-08], '[ 0.35836  -0.175699 -0.005238]': [9.2643591187435703e-08], '[-0.507423 -0.507984 -0.094952]': [5.2774585021508891e-08], '[-0.344701 0.338921 0.033313]': [2.0506655751539871e-06], '[0.016844 -0.508771 0.094208]': [3.7555935471222934e-06], '[0.492995 0.491596 0.094956]': [7.4254546359453499e-06], '[-0.338517  0.345936 -0.033014]': [2.7901524340956385e-08], '[-0.180072 -0.179205 -0.005416]': [2.1801865979220736e-08], '[ 0.34551  -0.006347  0.034031]': [1.3187673206238593e-07], '[ 0.338955 -0.344642 -0.033263]': [1.1229381335908784e-07], '[0.338517 -0.345936 0.033014]': [3.5142918997366213e-06], '[ 0.016879  0.485697 -0.094325]': [7.996020182088337e-08], '[ 0.492995 -0.508404  0.094956]': [6.6139643997709108e-08], '[-0.339136 -0.005772 -0.033433]': [6.3970294089665813e-08], '[-0.358257  0.175078 -0.005221]': [4.2230067359705359e-08], '[0.358257 -0.175078 0.005221]': [3.7260399735061789e-06], '[ 0.486367  0.01688   0.094333]': [7.1513668944978015e-08], '[-0.016844  0.508771 -0.094208]': [6.320062662101833e-08], '[-0.507005  0.491596  0.094956]': [8.2623585695440482e-08], '[ 0.006277 -0.34548   0.033989]': [4.7847954903756929e-08], '[0.507423 0.507984 0.094952]': [4.5077906621060482e-06], '[-0.508062 0.016849 -0.094221]': [3.3789778068060716e-06], '[-0.175098  0.358254 -0.005226]': [6.169616947881984e-08], '[0.016844 0.491229 0.094208]': [3.7571603356822969e-06], '[-0.179294 -0.179983 0.005416]': [3.1078843576864967e-06]}
    candidates = 0
##    print ("Start Finding Weyl Points")
    for initial in ini:
##        print ('starting initial at {}'.format(initial))
        CANDIDATES,candidates,Rfft=find_min(initial,CANDIDATES,candidates,ef_index,HRaux,SRaux,read_S,alat,velkp1,nk1,nk2,nk3,bnd,nspin,a_vectors,b_vectors,v_k,snk1_range=[-0.5,0.5],snk2_range=[-0.5,0.5],snk3_range=[-0.5,0.5],npool=1,shift=0.0,nl=None,sh=None)
    band =bnd
   # for ispin in range(nspin):
   #     CANDIDATES,candidates=sym_test(CANDIDATES,candidates,HRaux,Rfft,band,b_vectors,ef_index,ispin)    
##    print (CANDIDATES) 
    WEYL = {}


    #plt.switch_backend('agg')
    model = tbmodels.Model.from_wannier_files(hr_file='z2pack_hamiltonian.dat')
    system = z2pack.tb.System(model,bands=32)

    def gap(k):
        eig = model.eigenval(k)
        return eig[32] - eig[21]


    candidates=0
    cart=[]
    for i in CANDIDATES.keys():
        cart.append(np.dot(list(map(float, i[1:-1].split( ))),b_vectors))
    for p in cart:
        for x in [-1,1]:
            for y in [-1,1]:
                for z in [-1,1]:
                    kq = np.dot([p[0]*x,p[1]*y,p[2]*z],np.linalg.inv(b_vectors))
                    result_1 = z2pack.surface.run(system=system,surface=z2pack.shape.Sphere(center=tuple(kq), radius=0.005))
                    invariant = z2pack.invariant.chern(result_1)
##                    print (tuple(kq))
##                    print ('Chirality at {};{}'.format(kq,invariant))
                    if invariant != 0:
                        new = True
                        for t in WEYL.keys():
                            if np.linalg.norm(np.asarray(kq)-list(map(float, t[1:-1].split( ))))<0.005:
                                new=False
                        if new:
                            candidates += 1
                            WEYL[str(kq).replace(",", "")]=invariant
##                            print ('found Weyl point No. {} at cartesian: {},  crystal: {} '.format(candidates,kq))
##    print (WEYL)
    if bool(WEYL):
        j = 1
        for k in WEYL.keys():
            print ('Found Candidate No. {} at {} with Chirality:{}'.format(j,k,WEYL[k]))
            j = j + 1

    else:
        print("No Candidate found.")
##    print (WEYL)  
##    print (cart)


def sym_test(Candidates,NewCandidates,candidates,HRaux,Rfft,band,b_vectors,ef_index,ispin):
    print ('starting sym test')
    for i in NewCandidates.keys():
        p=list(map(float, i[1:-1].split( )))
        p=np.dot(p,b_vectors)
        for x in [-1,1]:
            for y in [-1,1]:
                for z in [-1,1]:
                    kq = np.dot([p[0]*x,p[1]*y,p[2]*z],np.linalg.inv(b_vectors))
                    egap = find_egap(HRaux,kq,Rfft,band,b_vectors,ef_index,ispin)
                    if egap<0.00001:
                        new = True
                        for t in Candidates.keys():
#                            print (t)
#                            print (kq)
                       #     try:
#                            print (list(map(float, t[1:-1].split( ))))
                            
                            if np.linalg.norm(np.asarray(kq)-list(map(float, t[1:-1].split( ))))<0.0001:
                                new = False
                       #     except:
                       #         print (list(map(float, t[1:-1].split(','))))
                            
                       #         if np.linalg.norm(np.asarray(kq)-list(map(float, t[1:-1].split(','))))<0.001:
                       #             new = False
                        if new:
                            candidates += 1
                            print ('Sym Candidate No.{} found at {} with gap:{} eV'.format(candidates,str(kq).replace(",", ""),egap))
                            Candidates[str(kq).replace(",", "")] = [egap]
    return Candidates,candidates





def find_min(initial,Candidates,candidates,ef_index,HRaux,SRaux,read_S,alat,velkp1,nk1,nk2,nk3,bnd,nspin,a_vectors,b_vectors,v_k,snk1_range=[-0.5,0.5],snk2_range=[-0.5,0.5],snk3_range=[-0.5,0.5],npool=1,shift=0.0,nl=None,sh=None):
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
    guess_K = search_grid+np.array([initial[0]/snk1,initial[1]/snk2,initial[2]/snk3])


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

####   weyl=True


    NewCandidates = {}
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
#    if rank ==0:print (sg)
    comm.Barrier()

    fold_coord = lambda x: ((x+0.5)%1.0-0.5)

    extrema=[]
    timer_avg = 0.0

#    if rank==0:
#        print("starting find crit at grid %s"%sst)
    fp=0
    sp=0
    ep=0
    
    c = -1
    for i in range(sg.shape[0]//size+1):

        c+=1
#        c = next(counter)

        if c>=sg.shape[0]:
            continue

        i     = sg[c][0]
        b     = sg[c][1]
        ispin = sg[c][2]
     #   print ('##########################')
     #   K = [-0.492995, -0.015409, -0.094956]
     #   K = [-0.491938, -0.016849, -0.094221]
     #   gap = find_egap(HRaux,K,Rfft,band,b_vectors,ef_index,ispin)
     #   print ('gap at [-0.491938, -0.016849, -0.094221] : {} eV'.format(gap))
     #   exit()
####        if Weyl:
        if True:
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
            x0 = np.asarray(guess_K[i]).ravel()
            n, = x0.shape
##            print ('bounds_K[i]:{}',format(bounds_K[i]))
##            print ('length of x0:{}, length of bounds:{}'.format(n,len(bounds_K[i])))
            current_bounds=[(bounds_K[i,0,0],bounds_K[i,1,0]),(bounds_K[i,0,1],bounds_K[i,1,1]),(bounds_K[i,0,2],bounds_K[i,1,2])]
##            print ('current_bounds:',format(current_bounds))
            solx = OP.fmin_l_bfgs_b(lam_XiP,guess_K[i],bounds=current_bounds,approx_grad=True,maxiter=3000)
#                                    method='trf',jac='2-point',ftol=9.9e-3,
#                                    max_nfev=300)
#            print ('solx:{}'.format(solx))

#            print (solx[1])
##            nfvf = solx.nfev
            #second pass
#            if np.all(np.abs(solx[1])<0.01):
#                print ('starting second pass')
            if np.abs(solx[1]<0.00001):
               # print ('Candidate No.{} found'.format(candidates))
               # print (solx[0])
               # print (solx[1])
               # print type(solx[0])
                if len(Candidates.keys()) == 0:
                    Candidates[str(solx[0])] = [solx[1]]
                    NewCandidates[str(solx[0])] = [solx[1]]
                    candidates += 1
  ##                  print ('Candidate No.{} found at {} with gap:{} eV'.format(candidates,solx[0],solx[1]))
                else:
                    real = True
                    for i in Candidates.keys():
                        if np.linalg.norm(solx[0]-list(map(float, i[1:-1].split( ))))<0.0001:
                            real = False
                    if real:
                        candidates += 1
   ##                     print ('Candidate No.{} found at {} with gap:{} eV'.format(candidates,solx[0],solx[1]))
                        Candidates[str(solx[0])] = [solx[1]]
                        NewCandidates[str(solx[0])] = [solx[1]]
   ##                 else:
   ##                     print ('Duplicate found at {} with gap:{} eV'.format(solx[0],solx[1]))
                     

#                solx = OP.least_squares(lam_XiP,solx.x,bounds=bounds_K[i],jac='3-point',max_nfev=30000)
#ftol=5.e-16,gtol=1.e-14,
#                                        method='trf',jac='3-point',xtol=1.e-12,
#                                        max_nfev=300)
            else:
                fp+=1
                continue


 #       except Exception,e:
 #           print(e)                   
 #           continue

 #       Candidates,candidates=sym_test(Candidates,NewCandidates,candidates,HRaux,Rfft,band,b_vectors,ef_index,ispin)  

    comm.Barrier()
 #   all_extrema = Gatherv_wrap(extrema)

#        zero_mask   = np.sum(all_extrema,axis=1)!=0.0
#        all_extrema = all_extrema[zero_mask]
#        if weyl:
#            np.savetxt('weyl.dat',all_extrema)

#    Candidates,candidates=sym_test(NewCandidates,candidates,HRaux,Rfft,band,b_vectors,ef_index,ispin)  
    return (Candidates,candidates,Rfft)
####    if weyl:
####        raise SystemExit

    comm.Barrier() 
    tempt = np.asarray([timer_avg],dtype=float)
    tottt = np.zeros((1),dtype=float)
    comm.Reduce(tempt,tottt)

    temp = np.asarray([ep,fp,sp],dtype=int)
    tot = np.zeros((3),dtype=int)
    comm.Reduce(temp,tot)
    if rank==0:
        fprm  = 1.0-float(tot[1])/float(tot[0])
        sprm = 1.0-float(tot[2])/float(tot[1])

        print("1st Pass: %s"%(np.around(fprm*100,decimals=2)))
        print("2nd Pass: %s"%(np.around(sprm*100,decimals=2)))
        print("AVG time: %s"%(tottt/float(tot[0])))


    if rank==0:
        return all_extrema
    else: return

"""







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
"""
