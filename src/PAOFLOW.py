# -*- coding: utf-8 -*-
# *************************************************************************************
# *                                                                                   *
# *   PAOFLOW *  Marco BUONGIORNO NARDELLI * University of North Texas 2016-2018      *
# *                                                                                   *
# *************************************************************************************
#
#  Copyright 2016-2018 - Marco BUONGIORNO NARDELLI (mbn@unt.edu) - AFLOW.ORG consortium
#
#  This file is part of AFLOW software.
#
#  AFLOW is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# *************************************************************************************

# future imports


# import general modules
import os, sys, traceback, psutil, time
from scipy import fftpack as FFT
from scipy import linalg as LA
from numpy import linalg as LAN
import xml.etree.ElementTree as ET
import numpy as np
#import numexpr as ne
from mpi4py import MPI
import multiprocessing

# Define paths
sys.path.append(sys.path[0]+'/defs')

# Import PAO specific functions
from build_Pn import *
from build_Hks import *
from do_non_ortho import *
from do_ortho import *
from add_ext_field import *
from get_R_grid_fft import *
from get_K_grid_fft import *
from do_bands_calc import *
from do_bands_calc_1D import *
from do_double_grid import *
from do_spin_orbit import *
from constants import *
try:
    from cuda_fft import *
except: pass
from read_inputfile_xml_parse import *
from read_QE_output_xml_parse import *
from read_new_QE_output_xml_parse import *
from write3Ddatagrid import *
from plot_compare_PAO_DFT_eigs import *
from do_topology_calc_new import *
from calc_PAO_eigs_vecs import *
from do_dos_calc import *
from do_pdos_calc import *
from do_fermisurf import *
from do_spin_texture import *
from do_gradient import *
from do_momentum import *
from do_dos_calc_adaptive import *
from do_pdos_calc_adaptive import *
from do_spin_Berry_curvature import *
from do_spin_Hall_conductivity import *
from do_spin_current import *
from write2bxsf import *
from do_Berry_curvature import *
from do_Berry_conductivity import *
from write2bxsf import *
from do_Boltz_tensors import *
from do_epsilon import *
from do_adaptive_smearing import *
from do_z2pack import *
import resource
from do_d2Ed2k import *
from do_Rij import *
from get_degeneracies import *
from do_perturb_split import *

def paoflow(inputpath='./',inputfile='inputfile.xml'):
    try:
        #----------------------
        # initialize parallel execution
        #----------------------
        comm=MPI.COMM_WORLD
        size=comm.Get_size()
    
        #----------------------
        # initialize time
        #----------------------
        if rank == 0: start = time.time()

        #----------------------
        # Print header
        #----------------------
        if rank == 0:
            print('          ')
            print('#############################################################################################')
            print('#                                                                                           #')
            print('#                                          PAOFLOW                                          #')
            print('#                                                                                           #')
            print('#                  Utility to construct and operate on Hamiltonians from                    #')
            print('#                 the Projections of DFT wfc on Atomic Orbital bases (PAO)                  #')
            print('#                                                                                           #')
            print('#                         ©2016-2018 ERMES group (http://ermes.unt.edu)                     #')
            print('#############################################################################################')
            print('          ')
    
        #----------------------
        # Initialize n. of threads for multiprocessing (FFTW)
        #----------------------
        nthread = multiprocessing.cpu_count()
        #ne.set_num_threads(nthread)
    except Exception as e:
        print('Rank %d: Exception in Initialization'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    
    try:
        #----------------------
        # Read input
        #----------------------
        fpath,restart,verbose,non_ortho,write2file,write_binary,writedata,writez2pack,use_cuda,shift_type, \
        shift,pthr,npool,do_comparison,naw,sh,nl,Efield,Bfield,HubbardU,bval,onedim,do_bands, \
        ibrav,dkres,nk,band_topology,spol,ipol,jpol,do_spin_orbit,theta,phi,lambda_p,lambda_d, \
        double_grid,nfft1,nfft2,nfft3,do_dos,do_pdos,emin,emax,delta,smearing,fermisurf, \
        fermi_up,fermi_dw,spintexture,d_tensor,t_tensor,a_tensor,s_tensor,temp,Boltzmann, \
        epsilon,metal,kramerskronig,epsmin,epsmax,ne,critical_points,Berry,eminAH,emaxAH, \
        ac_cond_Berry,spin_Hall,eminBT,emaxBT,eminSH,emaxSH,ac_cond_spin,eff_mass,tmin,tmax,tstep,\
        carrier_conc, out_vals = read_inputfile_xml(inputpath,inputfile)
        
        if double_grid:
            if nfft1%2!=0 or nfft2%2!=0 or nfft3%2!=0:
                nfft1 = 2*((nfft1+1)//2)
                nfft2 = 2*((nfft2+1)//2)
                nfft3 = 2*((nfft3+1)//2)
                print('Warning: nfft grid has been modified to support double_grid,')
                print('modified nfft grid:{} {} {}\n'.format(nfft1,nfft2,nfft3))
       
        fpath = os.path.join(inputpath, fpath)

        #----------------------
        # initialize return dictionary
        #----------------------
        if rank == 0 and out_vals is not None:
            outDict = {}
            if len(out_vals) > 0:
                for i in out_vals:
                    outDict[i] = None
 
        #----------------------
        # Check for fftw libraries
        #----------------------
        scipyfft = True
        if use_cuda:
            scipyfft = False
    
        #----------------------
        # Do dimension checks
        #----------------------
    
        comm.Barrier()
    except Exception as e:
        print('Rank %d: Exception in FFT Library or Dimension Check'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    
    try:
        #----------------------
        # Read DFT data
        #----------------------
    
        # Initialize variables
        U = None
        my_eigsmat = None
        alat = None
        a_vectors = b_vectors = None
        nkpnts = None
        nspin = None
        dftSO = None
        kpnts_wght = None
        nelec = None
        nbnd = None
        Efermi = None
        nawf = None
        nk1 = nk2 = nk3 = None
        natoms = None
        tau = None
        Sks = None

        if rank == 0 :

            if os.path.exists(fpath+'/data-file.xml'):
                if (not non_ortho):
                    U,my_eigsmat,alat,a_vectors,b_vectors, \
                    nkpnts,nspin,dftSO,kpnts,kpnts_wght, \
                    nelec,nbnds,Efermi,nawf,nk1,nk2,nk3,natoms,tau  =  read_QE_output_xml(fpath, verbose, non_ortho)

                    sumk = np.sum(kpnts_wght)
                    kpnts_wght /= sumk

                    if verbose: print('...using orthogonal algorithm')
                else:
                    U,Sks,my_eigsmat,alat,a_vectors,b_vectors, \
                    nkpnts,nspin,dftSO,kpnts,kpnts_wght, \
                    nelec,nbnds,Efermi,nawf,nk1,nk2,nk3,natoms,tau  =  read_QE_output_xml(fpath,verbose,non_ortho)
                    if verbose: print('...using non-orthogonal algorithm')
            elif os.path.exists(fpath+'/data-file-schema.xml'):
                if (not non_ortho):
                    U,my_eigsmat,alat,a_vectors,b_vectors, \
                    nkpnts,nspin,dftSO,kpnts,kpnts_wght, \
                    nelec,nbnds,Efermi,nawf,nk1,nk2,nk3,natoms,tau  =  read_new_QE_output_xml(fpath, verbose, non_ortho)

                    sumk = np.sum(kpnts_wght)
                    kpnts_wght /= sumk

                    if verbose: print('...using orthogonal algorithm')
                else:
                    U,Sks,my_eigsmat,alat,a_vectors,b_vectors, \
                    nkpnts,nspin,dftSO,kpnts,kpnts_wght, \
                    nelec,nbnds,Efermi,nawf,nk1,nk2,nk3,natoms,tau  =  read_new_QE_output_xml(fpath,verbose,non_ortho)
                    if verbose: print('...using non-orthogonal algorithm')

        # Broadcast data
        alat = comm.bcast(alat,root=0)
        a_vectors = comm.bcast(a_vectors,root=0)
        b_vectors = comm.bcast(b_vectors,root=0)
        nkpnts = comm.bcast(nkpnts,root=0)
        nspin = comm.bcast(nspin,root=0)
        dftSO = comm.bcast(dftSO,root=0)
        kpnts_wght = comm.bcast(kpnts_wght,root=0)
        nelec = comm.bcast(nelec,root=0)
        nbnd = comm.bcast(nbnd,root=0)
        Efermi = comm.bcast(Efermi,root=0)
        nawf = comm.bcast(nawf,root=0)
        nk1 = comm.bcast(nk1,root=0)
        nk2 = comm.bcast(nk2,root=0)
        nk3 = comm.bcast(nk3,root=0)
        natoms = comm.bcast(natoms,root=0)
        tau = comm.bcast(tau,root=0)

        #set npool to minimum needed if npool isnt high enough
        int_max = 2147483647
        temp_pool = int(np.ceil((float(nawf**2*nfft1*nfft2*nfft3*3*nspin)/float(int_max))))
        if temp_pool>npool:
            if rank==0:
                print("Warning: %s too low. Setting npool to %s"%(npool,temp_pool))
            npool = temp_pool

        if size >  1:
            if rank == 0 and npool == 1: print('parallel execution on {} processors, {} threads and {} pool'.format(size,nthread,npool))
            if rank == 0 and npool > 1: print('parallel execution on {} processors, {} threads and {} pools'.format(size,nthread,npool))
        else:
            if rank == 0: print('serial execution')

        #----------------------
        # Do memory checks 
        #----------------------

        if rank == 0:
            gbyte = nawf**2*nfft1*nfft2*nfft3*3*2*16./1.E9
            print('estimated maximum array size: %5.2f GBytes' %(gbyte))
            print('   ')

        comm.Barrier()
        if rank == 0:
            print('reading in                       %5s sec ' %str('%.3f' %(time.time()-start)).rjust(10))
            reset=time.time()
    except Exception as e:
        print('Rank %d: Exception in Reading DFT Data'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    
    try:
        #----------------------
        # Building the Projectability
        #----------------------
        bnd = None
        if rank != 0: shift = None
        if rank == 0:
            Pn = build_Pn(nawf,nbnds,nkpnts,nspin,U)
    
            if verbose: print('Projectability vector ',Pn)
    
            # Check projectability and decide bnd
    
            bnd = 0
            for n in range(nbnds):
                if Pn[n] > pthr:
                    bnd += 1
            Pn = None
            if verbose: print('# of bands with good projectability > {} = {}'.format(pthr,bnd))
            if verbose and bnd < nbnds: print('Range of suggested shift ',np.amin(my_eigsmat[bnd,:,:]),' , ', \
                                            np.amax(my_eigsmat[bnd,:,:]))
            if shift == 'auto': shift = np.amin(my_eigsmat[bnd,:,:])

    
        # Broadcast 
        bnd = comm.bcast(bnd,root=0)
        shift = comm.bcast(shift,root=0)

        emaxAH = np.amin(np.array([shift,emaxAH]))
        emaxSH = np.amin(np.array([shift,emaxSH]))
        emax = np.amin(np.array([shift,emax]))

    except Exception as e:
        print('Rank %d: Exception in Building Projectability'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    
    try:
        #----------------------
        # Building the PAO Hamiltonian
        #----------------------
    
        Hks = None
        if rank == 0:
            Hks = build_Hks(nawf,bnd,nkpnts,nspin,shift,my_eigsmat,shift_type,U)
            # This is needed for consistency of the ordering of the matrix elements
            # Important in ACBN0 file writing
            if non_ortho:
                Sks = np.transpose(Sks,(1,0,2))

        #U not needed anymore
        U = None

        comm.Barrier()
        if rank == 0:
            print('building Hks in                  %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
            reset=time.time()
            nawf = Hks.shape[0]
    except Exception as e:
        print('Rank %d: Exception in Building PAO Hamiltonian'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    
    # NOTE: Take care of non-orthogonality, if needed
    # Hks from projwfc is orthogonal. If non-orthogonality is required, we have to 
    # apply a basis change to Hks as Hks -> Sks^(1/2)+*Hks*Sks^(1/2)
    # non_ortho flag == 0 - makes H non orthogonal (original basis of the atomic pseudo-orbitals)
    # non_ortho flag == 1 - makes H orthogonal (rotated basis) 
    #    Hks = do_non_ortho(Hks,Sks)
    #    Hks = do_ortho(Hks,Sks)

    try:
        if rank == 0 and non_ortho:
            Hks = do_non_ortho(Hks,Sks)
    except Exception as e:
        print('Rank %d: Exception in Orthogonality'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    
    try:
        if rank == 0 and write2file:
            #----------------------
            # write to file Hks,Sks,kpnts,kpnts_wght
            #----------------------                                                                                                                           
            if write_binary:# or whatever you want to call it            
                if nspin==1:#postfix .npy just to make it clear what they are
                    np.save('kham.npy',np.ravel(Hks[...,0],order='C'))
                if nspin==2:
                    np.save('kham_up.npy',np.ravel(Hks[...,0],order='C'))
                    np.save('kham_dn.npy',np.ravel(Hks[...,1],order='C'))
                if non_ortho:
                    np.save('kovp.npy',np.ravel(Sks,order='C'))
            else:
                if nspin == 1:
                    f=open(os.path.join(inputpath,'kham.txt'),'w')
                    for ik in range(nkpnts):
                        for i in range(nawf):
                            for j in range(nawf):
                                f.write('%20.13f %20.13f \n' %(np.real(Hks[i,j,ik,0]),np.imag(Hks[i,j,ik,0])))
                    f.close()
                elif nspin == 2:
                    f=open(os.path.join(inputpath,'kham_up.txt'),'w')
                    for ik in range(nkpnts):
                        for i in range(nawf):
                            for j in range(nawf):
                                f.write('%20.13f %20.13f \n' %(np.real(Hks[i,j,ik,0]),np.imag(Hks[i,j,ik,0])))
                    f.close()
                    f=open(os.path.join(inputpath,'kham_down.txt'),'w')
                    for ik in range(nkpnts):
                        for i in range(nawf):
                            for j in range(nawf):
                                f.write('%20.13f %20.13f \n' %(np.real(Hks[i,j,ik,1]),np.imag(Hks[i,j,ik,1])))
                    f.close()
                if non_ortho:
                    f=open(os.path.join(inputpath,'kovp.txt'),'w')
                    for ik in range(nkpnts):
                        for i in range(nawf):
                            for j in range(nawf):
                                f.write('%20.13f %20.13f \n' %(np.real(Sks[i,j,ik]),np.imag(Sks[i,j,ik])))
                    f.close()
            f=open(os.path.join(inputpath,'k.txt'),'w')
            for ik in range(nkpnts):
                f.write('%20.13f %20.13f %20.13f \n' %(kpnts[ik,0],kpnts[ik,1],kpnts[ik,2]))
            f.close()
            f=open(os.path.join(inputpath,'wk.txt'),'w')
            for ik in range(nkpnts):
                f.write('%20.13f \n' %(kpnts_wght[ik]))
            f.close()
    
            print('H(k),S(k),k,wk written to file')
        if write2file: quit()
    except Exception as e:
        print('Rank %d: Exception in Write to File'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    
    try:
        #----------------------
        # Plot the PAO and DFT eigevalues. Writes to comparison.pdf
        #----------------------
        if rank == 0 and do_comparison:
            plot_compare_PAO_DFT_eigs(Hks,Sks,my_eigsmat,non_ortho)
            quit()
    except Exception as e:
        print('Rank %d: Exception in Do Comparison'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    
    my_eigsmat=None
    try:
        #----------------------
        # Define the Hamiltonian and overlap matrix in real space: HRs and SRs (noinv and nosym = True in pw.x)
        #----------------------
        HRs = SRs = None
        if rank ==0:
            # Original k grid to R grid
            reset=time.time()
            Hkaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
            Skaux  = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex)

            Hkaux = np.reshape(Hks,(nawf,nawf,nk1,nk2,nk3,nspin),order='C')
            if non_ortho:
                Skaux = np.reshape(Sks,(nawf,nawf,nk1,nk2,nk3),order='C')
    
            HRaux = np.zeros_like(Hkaux)
            SRaux = np.zeros_like(Skaux)
    
            if use_cuda:
                HRaux = np.moveaxis(cuda_ifftn(np.moveaxis(Hkaux,[0,1],[3,4]),axes=[0,1,2]),[3,4],[0,1])
            else:
                HRaux = FFT.ifftn(Hkaux,axes=[2,3,4])
    
            if non_ortho:
                if use_cuda:
                    SRaux = np.moveaxis(cuda_ifftn(np.moveaxis(Hkaux,[0,1],[3,4]),axes=[0,1,2]),[3,4],[0,1])
                else:
                    SRaux = FFT.ifftn(Skaux,axes=[2,3,4])
    
            # NOTE: Naming convention (from here):
            # Hks = k-space Hamiltonian on original MP grid
            # HRs = R-space Hamiltonian on original MP grid
            ############################
            ####  IS THIS NEEDED??  ####
            ############################

            Hks = None
            Sks = None
            Hks = Hkaux
            Sks = Skaux
            HRs = HRaux
            SRs = SRaux
            Hkaux = None
            Skaux = None
            HRaux = None
            SRaux = None
    
        comm.Barrier()
        if rank == 0:
            print('k -> R in                        %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
            reset=time.time()
    except Exception as e:
        print('Rank %d: Exception in k->R'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    
    try:
        if rank == 0 and (Efield.any() != 0.0 or Bfield.any() != 0.0 or HubbardU.any() != 0.0):
            # Add external fields or non scf ACBN0 correction
            tau_wf = np.zeros((nawf,3),dtype=float)
            l=0
            for n in range(natoms):
                for i in range(naw[n]):
                    tau_wf[l,:] = tau[n,:]
                    l += 1
    
            # Define real space lattice vectors
            R,Rfft,R_wght,nrtot,idx = get_R_grid_fft(nk1,nk2,nk3,a_vectors)
    
            HRs = add_ext_field(HRs,tau_wf,R,alat,Efield,Bfield,HubbardU)
    except Exception as e:
        print('Rank %d: Exception in Adding External Fields'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    
    try:
        if rank == 0 and do_spin_orbit:
            #----------------------
            # Compute bands with spin-orbit coupling
            #----------------------
    
            socStrengh = np.zeros((natoms,2),dtype=float)
            socStrengh [:,0] =  lambda_p[:]
            socStrengh [:,1] =  lambda_d[:]
    
            HRs = do_spin_orbit_calc(HRs,natoms,theta,phi,socStrengh)
            nawf=2*nawf
    except Exception as e:
        print('Rank %d: Exception in Do Spin Orbit'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception

    try:
        if rank == 0 and writez2pack:
            do_z2pack_hamiltonian(nawf,nk1,nk2,nk3,a_vectors,HRs)
    except Exception as e:
        print('Rank %d: Exception in Do Z2pack'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    

    if do_bands and not(onedim):
            #----------------------
            # Compute bands on a selected path in the BZ
            #----------------------
    
            alat /= ANGSTROM_AU
    
            if rank == 0 and non_ortho:
                # now we orthogonalize the Hamiltonian again
                if use_cuda:
                    Hkaux  = np.zeros((nk1,nk2,nk3,nawf,nawf,nspin),dtype=complex)
                    Hkaux[:,:,:,:,:,:] = cuda_fftn(np.moveaxis(HRs,[0,1],[3,4]),axes=[0,1,2])
                    Skaux  = np.zeros((nk1,nk2,nk3,nawf,nawf),dtype=complex)
                    Skaux[:,:,:,:,:,:] = cuda_fftn(np.moveaxis(SRs,[0,1],[3,4]),axes=[0,1,2])
                    Hkaux = np.reshape(np.moveaxis(Hkaux,[3,4],[0,1]),(nawf,nawf,nk1*nk2*nk3,nspin),order='C')
                    Skaux = np.reshape(np.moveaxis(Skaux,[3,4],[0,1]),(nawf,nawf,nk1*nk2*nk3),order='C')
                else:
                    Hkaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
                    Hkaux[:,:,:,:,:,:] = FFT.fftn(HRs[:,:,:,:,:,:],axes=[2,3,4])
                    Skaux  = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex)
                    Skaux[:,:,:,:,:] = FFT.fftn(SRs[:,:,:,:,:],axes=[2,3,4])
                    Hkaux = np.reshape(Hkaux,(nawf,nawf,nk1*nk2*nk3,nspin),order='C')
                    Skaux = np.reshape(Skaux,(nawf,nawf,nk1*nk2*nk3),order='C')
    
                Hkaux = do_ortho(Hkaux,Skaux)
                Hkaux = np.reshape(Hkaux,(nawf,nawf,nk1,nk2,nk3,nspin),order='C')
                Skaux = np.reshape(Skaux,(nawf,nawf,nk1,nk2,nk3),order='C')
                if use_cuda:
                    HRs = np.moveaxis(cuda_ifftn(np.moveaxis(Hkaux,[0,1],[3,4]),axes=[0,1,2]),[3,4],[0,1])
                else:
                    HRs[:,:,:,:,:,:] = FFT.ifftn(Hkaux[:,:,:,:,:,:],axes=[2,3,4])
    
            non_ortho = False
    
            # Broadcast HRs and SRs
            if rank!=0:
                HRs=np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex,order='C')
            comm.Bcast(np.ascontiguousarray(HRs),root=0)

            if non_ortho: 
                if rank!=0:
                    SRs=np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex,order='C')
                comm.Bcast(SRs,root=0)
            else: 
                SRs = None
                Sks = None


            # Define real space lattice vectors
            R,Rfft,R_wght,nrtot,idx = get_R_grid_fft(nk1,nk2,nk3,a_vectors)
    
            # Define k-point mesh for bands interpolation
            kq = kpnts_interpolation_mesh(ibrav,alat,a_vectors,b_vectors,nk,inputpath)
            nkpi=kq.shape[1]
            for n in range(nkpi):
                kq[:,n]=np.dot(kq[:,n],b_vectors)
    
            # Compute the bands along the path in the IBZ
            E_kp = v_kp = None

            E_kp,v_kp = do_bands_calc(HRs,SRs,kq,R_wght,R,idx,non_ortho,inputpath,npool)

            if rank == 0:
                print('bands in                         %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
                reset=time.time()
    

            if band_topology:
                # Compute Z2 invariant, velocity, momentum and Berry curvature and spin Berry
                # curvature operators along the path in the IBZ from do_topology_calc 
                eff_mass=True
                do_topology_calc(HRs,SRs,non_ortho,kq,E_kp,v_kp,R,Rfft,R_wght,idx,alat,b_vectors,nelec,bnd,Berry,ipol,jpol,spin_Hall,spol,do_spin_orbit,sh,nl,eff_mass,inputpath,npool)

                comm.Barrier()
                if rank == 0:
                    print('band topology in                 %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
                    reset=time.time()
    
            alat *= ANGSTROM_AU
    
    elif do_bands and onedim:
            #----------------------
            # FFT interpolation along a single directions in the BZ
            #----------------------
            if rank == 0 and verbose: print('... computing bands along a line')
            if rank == 0: do_bands_calc_1D(Hks,inputpath)
    
            comm.Barrier()
            if rank ==0:
                print('bands in                          %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
                reset=time.time()

    E_kp = v_kp = None

    try:
        #----------------------
        # Initialize Read/Write restart data
        #----------------------
        checkpoint = 0

        if restart:
            try:
                datadump = np.load(fpath+'PAOdump_%s.npz'%rank)
                checkpoint = datadump['checkpoint']
                if rank == 0:
                    print('reading data from dump at checkpoint ',checkpoint)
            except:
                pass
        checkpoint = comm.bcast(checkpoint,root=0)
    except Exception as e:
        print('Rank %d: Exception in Checkpoint Initialization'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    
    try:
        #----------------------
        # Start master-slaves communication
        #----------------------
        Hksp = None
        if rank == 0:
            if double_grid and checkpoint == 0:
    
                if non_ortho:
                    # now we orthogonalize the Hamiltonian again
                    if use_cuda:
                        Hkaux  = np.zeros((nk1,nk2,nk3,nawf,nawf,nspin),dtype=complex)
                        Hkaux[:,:,:,:,:,:] = cuda_fftn(np.moveaxis(HRs,[0,1],[3,4]),axes=[0,1,2])
                        Skaux  = np.zeros((nk1,nk2,nk3,nawf,nawf),dtype=complex)
                        Skaux[:,:,:,:,:,:] = cuda_fftn(np.moveaxis(SRs,[0,1],[3,4]),axes=[0,1,2])
                        Hkaux = np.reshape(np.moveaxis(Hkaux,[3,4],[0,1]),(nawf,nawf,nk1*nk2*nk3,nspin),order='C')
                        Skaux = np.reshape(np.moveaxis(Skaux,[3,4],[0,1]),(nawf,nawf,nk1*nk2*nk3),order='C')
                    else:
                        Hkaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
                        Hkaux[:,:,:,:,:,:] = FFT.fftn(HRs[:,:,:,:,:,:],axes=[2,3,4])
                        Skaux  = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex)
                        Skaux[:,:,:,:,:] = FFT.fftn(SRs[:,:,:,:,:],axes=[2,3,4])
                        Hkaux = np.reshape(Hkaux,(nawf,nawf,nk1*nk2*nk3,nspin),order='C')
                        Skaux = np.reshape(Skaux,(nawf,nawf,nk1*nk2*nk3),order='C')
        
                    Hkaux = do_ortho(Hkaux,Skaux)
                    Hkaux = np.reshape(Hkaux,(nawf,nawf,nk1,nk2,nk3,nspin),order='C')
                    Skaux = np.reshape(Skaux,(nawf,nawf,nk1,nk2,nk3),order='C')
                    if use_cuda:
                        HRs = np.moveaxis(cuda_ifftn(np.moveaxis(Hkaux,[0,1],[3,4]),axes=[0,1,2]),[3,4],[0,1])
                    else:
                        HRs[:,:,:,:,:,:] = FFT.ifftn(Hkaux[:,:,:,:,:,:],axes=[2,3,4])

        non_ortho = False
        Skaux = None
        SRs   = None
        Sks   = None

        comm.Barrier()


        if double_grid:
            nr1=nk1
            nr2=nk2
            nr3=nk3
            #----------------------
            # Fourier interpolation on extended grid (zero padding)
            #----------------------
            Hksp,nk1,nk2,nk3 = do_double_grid(nfft1,nfft2,nfft3,HRs,nthread,npool)
            # Naming convention (from here): 
            # Hksp = k-space Hamiltonian on interpolated grid
            if rank == 0 and verbose: print('Grid of k vectors for zero padding Fourier interpolation ',nk1,nk2,nk3),
            kq,kq_wght,_,idk = get_K_grid_fft(nk1,nk2,nk3,b_vectors)
            if rank ==0:
                print('R -> k zero padding in           %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
                reset=time.time()    
        else:
            kq,kq_wght,_,idk = get_K_grid_fft(nk1,nk2,nk3,b_vectors)
            if rank == 0: Hksp = np.reshape(Hks,(nawf**2,nk1,nk2,nk3,nspin),order='C')
            Hksp=scatter_full(Hksp,npool)

        #no longer needed
        HRs = None
        Hkaux = None
        Hks   = None

        #----------------------
        # Read/Write restart data
        #----------------------
        
        if restart:
            if checkpoint == 0:
                checkpoint += 1
                np.savez(fpath+'PAOdump_%s.npz'%rank,checkpoint=checkpoint,Hksp=Hksp,kq=kq,kq_wght=kq_wght,idk=idk,nk1=nk1,nk2=nk2,nk3=nk3)
            elif checkpoint > 0:
                Hksp = datadump['Hksp']
                kq = datadump['kq']
                kq_wght = datadump['kq_wght']
                idk = datadump['idk']
                nk1 = datadump['nk1']
                nk2 = datadump['nk2']
                nk3 = datadump['nk3']
        else:
            if rank!=0:

                kq = None
                kq_wght = None
                idk = None
                nk1 = None
                nk2 = None
                nk3 = None

        checkpoint = comm.bcast(checkpoint,root=0)

    except Exception as e:
        print('Rank %d: Exception in Do Double Grid'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    
    try:
        #----------------------
        # Compute eigenvalues of the interpolated Hamiltonian
        #----------------------
        if checkpoint < 2:
    
            eig = None
            E_k = None
            v_k = None
            E_k, v_k = calc_PAO_eigs_vecs(Hksp,bnd,npool)

            # get the degenerate eigenvalues for each H(k)
            degen = get_degeneracies(E_k,bnd)

            if HubbardU.any() != 0.0:
                E_k = gather_full(E_k,npool)
                if rank==0:
                    E_k -= np.amax(E_k[:,bval,:])
                comm.Barrier()
                E_k = scatter_full(E_k,npool)                    
                eig   = np.reshape(E_k[:,:bnd],(E_k.shape[0]*bnd,nspin),order="C")            
    

            _,nk1,nk2,nk3,_ = Hksp.shape
            
    
            comm.Barrier()
            if rank ==0:
                print('eigenvalues in                   %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
                reset=time.time()
    
        #----------------------
        # Read/Write restart data
        #----------------------

        if restart:
            if checkpoint == 1:
                checkpoint += 1
                np.savez(fpath+'PAOdump_%s.npz'%rank,checkpoint=checkpoint,Hksp=Hksp,kq=kq,kq_wght=kq_wght,idk=idk,nk1=nk1,nk2=nk2,nk3=nk3, \
                        eig=eig,E_k=E_k,v_k=v_k)
            elif checkpoint > 1:
                Hksp = datadump['Hksp']
                kq = datadump['kq']
                kq_wght = datadump['kq_wght']
                idk = datadump['idk']
                nk1 = datadump['nk1']
                nk2 = datadump['nk2']
                nk3 = datadump['nk3']
                eig = datadump['eig']
                E_k = datadump['E_k']
                v_k = datadump['v_k']
        else:
            if rank!=0:
                kq = None
                kq_wght = None
                idk = None
                nk1 = None
                nk2 = None
                nk3 = None
#                eig = None
#                E_k = None
#                v_k = None

        checkpoint = comm.bcast(checkpoint,root=0)
        nk1 = comm.bcast(nk1,root=0)
        nk2 = comm.bcast(nk2,root=0)
        nk3 = comm.bcast(nk3,root=0)
    except Exception as e:
        print('Rank %d: Exception in Eigenvalues'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    
    try:
        if (do_dos or do_pdos) and smearing == None:
            #----------------------
            # DOS calculation with gaussian smearing on double_grid Hksp
            #----------------------

            index = None
            if rank == 0:
                index = {'eigtot':nk1*nk2*nk3*bnd}
            index = comm.bcast(index,root=0)
            eigtot = index['eigtot']

            eigup = eigdw = None

            if nspin == 1 or nspin == 2:

                eigup = E_k[:,:bnd,0].reshape(E_k.shape[0]*bnd)
                do_dos_calc(eigup,emin,emax,delta,eigtot,bnd,0,inputpath,npool)
                eigup = None
            if nspin == 2:
                eigdw = E_k[:,:bnd,1].reshape(E_k.shape[0]*bnd)
                do_dos_calc(eigdw,emin,emax,delta,eigtot,bnd,1,inputpath,npool)
                eigdw = None

            if do_pdos:
                #----------------------
                # PDOS calculation
                #----------------------

                v_kup = v_kdw = None
                if nspin == 1 or nspin == 2:
                    eigup = E_k[:,:,0]
                    v_kup = v_k[:,:,:,0]
                    do_pdos_calc(eigup,emin,emax,delta,v_kup,nk1,nk2,nk3,nawf,0,inputpath)
                    eigup = None
                    v_kup = None
                if nspin == 2:
                    eigdw = E_k[:,:,1]
                    v_kdw = v_k[:,:,:,1]
                    do_pdos_calc(eigdw,emin,emax,delta,v_kdw,nk1,nk2,nk3,nawf,1,inputpath)
                    eigdw = None
                    v_kdw = None

            comm.Barrier()
            if rank ==0:
                print('dos in                           %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
                reset=time.time()
    except Exception as e:
        print('Rank %d: Exception in DOS Calculation'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    
    try:
        if fermisurf or spintexture:
            #----------------------
            # Fermi surface calculation
            #----------------------
    
            if nspin == 1 or nspin == 2:
                do_fermisurf(fermi_dw,fermi_up,E_k[:,:,0],alat,b_vectors,nk1,nk2,nk3,nawf,0,npool,inputpath)
                eigup = None
            if nspin == 2:                
                do_fermisurf(fermi_dw,fermi_up,E_k[:,:,1],alat,b_vectors,nk1,nk2,nk3,nawf,0,npool,inputpath)
                eigdw = None
            if spintexture and nspin == 1:
                do_spin_texture(fermi_dw,fermi_up,E_k,v_k,sh,nl,nk1,nk2,nk3,nawf,nspin,do_spin_orbit,npool,inputpath)
    

            if rank ==0:
                print('FermiSurf in                     %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
                reset=time.time()
    except Exception as e:
        print('Rank %d: Exception in Fermi Surface'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception


    
    if Boltzmann and carrier_conc:#do d2H/d2k_ij
        M_ij = do_d2Ed2k_ij(Hksp,a_vectors,alat,nthread,npool,use_cuda,v_k,bnd,degen)

    try:

        pksp = None
        jksp = None
        if Boltzmann or epsilon or Berry or spin_Hall or critical_points or smearing != None:
            if checkpoint < 3:

                ###############################################################################
                #----------------------
                # Make sure fourier interpolated Hksp is hermitian
                #----------------------           

                Hksp = np.ascontiguousarray(np.reshape(Hksp,(Hksp.shape[0],nk1*nk2*nk3,nspin)))
                Hksp = gather_scatter(Hksp,1,npool)                

                Hksp = np.ascontiguousarray(np.reshape(Hksp,(nawf,nawf,Hksp.shape[1],nspin)))
                for ik in range(Hksp.shape[2]):
                    for ispin in range(Hksp.shape[3]):
                        Hksp[:,:,ik,ispin] = (np.conj(Hksp[:,:,ik,ispin].T) + \
                                                      Hksp[:,:,ik,ispin])/2.0


                Hksp = np.ascontiguousarray(np.reshape(Hksp,(nawf*nawf,Hksp.shape[2],nspin)))
                Hksp = np.ascontiguousarray(np.swapaxes(Hksp,0,1))
                Hksp = gather_scatter(Hksp,1,npool)                
                Hksp = np.swapaxes(Hksp,0,1)

                Hksp = np.ascontiguousarray(np.reshape(Hksp,(Hksp.shape[0],nk1,nk2,nk3,nspin)))
                ###############################################################################
                #----------------------
                # Compute the gradient of the k-space Hamiltonian
                #----------------------           

                dHksp = do_gradient(Hksp,a_vectors,alat,nthread,npool,use_cuda)                

                #############################################################
                ################DISTRIBUTE ARRAYS ON KPOINTS#################
                #############################################################
                
                Hksp  = None
                dHksp = np.reshape(dHksp,(dHksp.shape[0],nk1*nk2*nk3,3,nspin))
                #gather dHksp on nawf*nawf and scatter on k points
                dHksp = gather_scatter(dHksp,1,npool)
                dHksp = np.rollaxis(dHksp,0,3)
                dHksp = np.reshape(dHksp,(dHksp.shape[0],3,nawf,nawf,nspin),order="C")



                if rank == 0:
                    print('gradient in                      %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
                    reset=time.time()
        
        #----------------------
        # Read/Write restart data
        #----------------------

        if restart:
            if checkpoint == 2:
                checkpoint += 1
                np.savez(fpath+'PAOdump_%s.npz'%rank,checkpoint=checkpoint,Hksp=Hksp,kq=kq,kq_wght=kq_wght,idk=idk,nk1=nk1,nk2=nk2,nk3=nk3, \
                        eig=eig,E_k=E_k,v_k=v_k,dHksp=dHksp)
            elif checkpoint > 2:
                Hksp = datadump['Hksp']
                kq = datadump['kq']
                kq_wght = datadump['kq_wght']
                idk = datadump['idk']
                nk1 = datadump['nk1']
                nk2 = datadump['nk2']
                nk3 = datadump['nk3']
                eig = datadump['eig']
                E_k = datadump['E_k']
                v_k = datadump['v_k']
                dHksp = datadump['dHksp']
        else:
            if rank!=0:
                Hksp = None
                kq = None
                kq_wght = None
                idk = None
                nk1 = None
                nk2 = None
                nk3 = None


        checkpoint = comm.bcast(checkpoint,root=0)
    
        if checkpoint < 4:
            #----------------------
            # Compute the momentum operator p_n,m(k) (and kinetic energy operator)
            #----------------------
            #from do_momentum_d2 
    
            if rank != 0:
                tksp = None

            pksp = do_momentum(v_k,dHksp)

            for ik in range(pksp.shape[0]):
                for l in range(pksp.shape[1]):
                    for ispin in range(pksp.shape[4]):
                        pksp[ik,l,:,:,ispin] = do_perturb_split(pksp[ik,l,:,:,ispin],
                                                                dHksp[ik,l,:,:,ispin],
                                                                v_k[ik,:,:,ispin],
                                                                degen[ispin][ik])



            if rank == 0:
                print('momenta in                       %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
                reset=time.time()
    
        #----------------------
        # Read/Write restart data
        #----------------------

        if restart:
            if checkpoint == 3:
                    checkpoint += 1
                    np.savez(fpath+'PAOdump_%s.npz'%rank,checkpoint=checkpoint,Hksp=Hksp,kq=kq,kq_wght=kq_wght,idk=idk,nk1=nk1,nk2=nk2,nk3=nk3, \
                            eig=eig,E_k=E_k,v_k=v_k,dHksp=dHksp,pksp=pksp)
            elif checkpoint > 3:
                Hksp = datadump['Hksp']
                kq = datadump['kq']
                kq_wght = datadump['kq_wght']
                idk = datadump['idk']
                nk1 = datadump['nk1']
                nk2 = datadump['nk2']
                nk3 = datadump['nk3']
                eig = datadump['eig']
                E_k = datadump['E_k']
                v_k = datadump['v_k']
                dHksp = datadump['dHksp']
                pksp = datadump['pksp']
        else:
            if rank!=0:
                Hksp = None
                kq = None
                kq_wght = None
                idk = None
                nk1 = None
                nk2 = None
                nk3 = None

        checkpoint = comm.bcast(checkpoint,root=0)
        nk1 = comm.bcast(nk1,root=0)
        nk2 = comm.bcast(nk2,root=0)
        nk3 = comm.bcast(nk3,root=0)

        index = None
        if rank == 0:
            index = {'nawf':E_k.shape[1],'nktot':nk1*nk2*nk3}
        index = comm.bcast(index,root=0)
        nawf = index['nawf']
        nktot = index['nktot']

        kq_wght = np.ones((nktot),dtype=float)
        kq_wght /= float(nktot)

    except Exception as e:
        print('Rank %d: Exception in Gradient or Momenta'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception




    if not spin_Hall:
        dHksp=None


    try:
        deltakp = None
        deltakp2 = None

        if smearing != None:
            deltakp,deltakp2 = do_adaptive_smearing(pksp,nawf,nspin,
                                                    alat,a_vectors,nk1,nk2,nk3,smearing)

            if restart:
                np.savez(fpath+'PAOdelta'+str(nspin)+'_%s.npz'%rank,
                         deltakp=deltakp,deltakp2=deltakp2)
    

        if rank == 0 and smearing != None:
            print('adaptive smearing in             %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
            reset=time.time()
    except Exception as e:
        print('Rank %d: Exception in Adaptive Smearing'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    
    # to test formula 23 in the Graf & Vogl paper...
    #if rank == 0:
    #    from smearing import *
    #    nawf = bnd
    #    Ef = 0.0
    #    effterm = np.zeros((nk1*nk2*nk3,nawf),dtype=complex)
    #    for n in xrange(nawf):
    #        for m in xrange(nawf):
    #            if m != n:
    #                fm = intgaussian(E_k[:,m,0],Ef,deltakp[:,m,0])
    #                effterm[:,n] += 1.0/(E_k[:,m,0]-E_k[:,n,0] + 1.0j*deltakp2[:,n,m,0]) * \
    #                                (pksp[:,ipol,n,m,0] * pksp[:,jpol,m,n,0] + pksp[:,jpol,n,m,0] * pksp[:,ipol,m,n,0])
    #
    #    f=open(inputpath+'Tnn'+str(LL[ipol])+str(LL[jpol])+'.dat','w')
    #    for n in xrange(nawf):
    #        f.write('%.5f %9.5e %9.5e \n' \
    #                %(n,np.sum(np.real(kq_wght[:]*tksp[:,ipol,jpol,n,n,0])), np.sum(np.real(kq_wght[:]*effterm[:,n]))))
    #    f.close()
    #
    #    Tsum = 0.0
    #    Psum = 0.0
    #    for n in xrange(nawf):
    #        fn = intgaussian(E_k[:,n,0],Ef,deltakp[:,n,0])
    #        Tsum += np.sum(np.real(kq_wght[:]*tksp[:,ipol,jpol,n,n,0])*fn)
    #        Psum += np.sum(np.real(kq_wght[:]*effterm[:,n])*fn)
    #    print (Tsum, Psum)
    #quit()
    
#    try:








    velkp = None
    if Boltzmann or critical_points:
        #----------------------
        # Compute velocities for Boltzmann transport
        #----------------------
        velkp = np.zeros((pksp.shape[0],3,bnd,nspin),dtype=float)
        for n in range(bnd):
            velkp[:,:,n,:] = np.real(pksp[:,:,n,n,:])

        if carrier_conc:
            #----------------------
            # for d2H/d2k_ij
            #----------------------
            ij_ind = np.array([[0,0],[1,1],[2,2],[0,1],[1,2],[0,2]],dtype=int)
            E_temp = np.zeros((bnd,nawf),order="C")

            for ispin in range(M_ij.shape[3]):
                for ik in range(M_ij.shape[1]):

                    E_temp = ((E_k[ik,:,ispin]-E_k[ik,:,ispin][:,None])[:,:])
                    E_temp[np.where(np.abs(E_temp)<1.e-6)]=np.inf

                    for ij in range(ij_ind.shape[0]):
                        ipol = ij_ind[ij,0]
                        jpol = ij_ind[ij,1]
                        temp = (pksp[ik,ipol,:,:,ispin]*pksp[ik,jpol,:,:,ispin].T + \
                                pksp[ik,jpol,:,:,ispin]*pksp[ik,ipol,:,:,ispin].T).real / E_temp
                        M_ij[ij,ik,:,ispin] += np.sum(temp,axis=0)[:bnd]

        if critical_points:
            velkp_full = gather_full(velkp,npool)

            if rank == 0:

                #----------------------
                # Find critical points (grad(E_kn)=0)
                #----------------------
                f=open(os.path.join(inputpath,'critical_points.dat'),'w')
                for ik in range(nk1*nk2*nk3):
                    for n in range(bnd):
                        for ipin in range(nspin):
                            if  np.abs(velkp_full[ik,0,n,ispin]) < 1.e-2 and \
                                np.abs(velkp_full[ik,1,n,ispin]) < 1.e-2 and \
                                np.abs(velkp_full[ik,2,n,ispin]) < 1.e-2:
                                f.write('band %5d at %.5f %.5f %.5f \n' %(n,kq[0,ik],kq[1,ik],kq[2,ik]))
                f.close()

            velkp_full = None

    try:
        if (do_dos or do_pdos) and smearing != None:
            #----------------------
            # DOS calculation with adaptive smearing on double_grid Hksp
            #----------------------
    
            index = None
            eigtot = nk1*nk2*nk3*bnd
        
            eigup = None
            eigdw = None
            deltakpup = None
            deltakpdw = None
            if do_dos:
                if nspin == 1 or nspin == 2:
                    deltakpup = np.ravel(deltakp[:,:bnd,0],order='C')
                    eigup = E_k[:,:bnd,0].reshape(E_k.shape[0]*bnd)
                    do_dos_calc_adaptive(eigup,emin,emax,deltakpup,eigtot,bnd,0,smearing,inputpath)
                    eigup = None
                    deltakpup = None
                if nspin == 2:
                    deltakpdw = np.ravel(deltakp[:,:bnd,0],order='C')
                    eigdw = E_k[:,:bnd,0].reshape(E_k.shape[0]*bnd)
                    do_dos_calc_adaptive(eigdw,emin,emax,deltakpdw,eigtot,bnd,1,smearing,inputpath)
                    eigdw = None
                    deltakpdw = None
        
            #no longer needed
            eig=None

            if do_pdos:
                v_kup = v_kdw = None
                #----------------------
                # PDOS calculation
                #----------------------    
                for ispin in range(nspin):
                    do_pdos_calc_adaptive(E_k[:,:,ispin],emin,emax,deltakp[:,:,ispin],v_k[:,:,:,ispin],
                                          nk1,nk2,nk3,nawf,ispin,smearing,inputpath)

     
            comm.Barrier()
            if rank ==0:
                print('dos (adaptive smearing) in       %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
                reset=time.time()
    except Exception as e:
        print('Rank %d: Exception in DOS (Adaptive Smearing)'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception

    #----------------------
    # Memory reduction
    #----------------------
    # Reduce memory requirements and improve performance by reducing nawf to bnd (states with good projectability)
    try:
        if rank==0:
            if 'E_k' in outDict:
                outDict['E_k'] = E_k
            if 'deltakp' in outDict:
                outDict['deltakp'] = deltakp

        if not spin_Hall:
            v_k = None

#        E_k = np.ascontiguousarray(E_k[:,:bnd])

#        if smearing != None:
#            deltakp = np.ascontiguousarray(deltakp[:,:bnd])
#            deltakp2 = np.ascontiguousarray(deltakp2[:,:bnd,:bnd])

#        pksp = np.ascontiguousarray(pksp[:,:,:bnd,:bnd])



    except Exception as e:
        print('Rank %d: Exception in Memory Reduction'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception


    try:
        #----------------------
        # Spin Hall calculation
        #----------------------
        if spin_Hall:
            if dftSO == False: sys.exit('full relativistic calculation with SO needed')
            for n in range(s_tensor.shape[0]):
                ipol = s_tensor[n][0]
                jpol = s_tensor[n][1]
                spol = s_tensor[n][2]
                #----------------------
                # Compute the spin current operator j^l_n,m(k)
                #----------------------
                jksp = None
                spincheck = 0
                if restart and rank == 0:
                    try:
                        spindump = np.load(fpath+'PAOspin'+str(spol)+'_%s.npz'%rank)
                        jksp = spindump['jksp']
                        spincheck += 1
                        print('reading spin current for polarization ',spol)
                    except:
                        pass
                spincheck=comm.bcast(spincheck,root=0)
                if spincheck == 0:
                    jksp = do_spin_current(v_k,dHksp,spol,npool,do_spin_orbit,sh,nl,bnd,degen)

                    if restart:
                        np.savez(fpath+'PAOspin'+str(spol)+'_%s.npz'%rank,jksp=jksp)
     

#                    if rank == 0:
#                        print('spin current in                  %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
#                        reset=time.time()
                #----------------------
                # Compute spin Berry curvature... 
                #----------------------

                ene,shc,Om_k = do_spin_Berry_curvature(E_k,jksp,pksp,nk1,nk2,nk3,npool,ipol,jpol,
                                                       eminSH,emaxSH,fermi_dw,fermi_up,deltakp,smearing)
                if writedata:
                    if rank == 0: 
                        Om_kps = np.zeros((nk1,nk2,nk3,2),dtype=float)
                        x0 = np.zeros(3,dtype=float)
                        ind_plot = np.zeros(2)
                        Om_kps[:,:,:,0] = Om_k
                        Om_kps[:,:,:,1] = Om_k
                        fname = 'spin_Berry_'+str(LL[spol])+'_'+str(LL[ipol])+str(LL[jpol])+'.bxsf'
                        write2bxsf(fermi_dw,fermi_up,Om_kps,nk1,nk2,nk3,2,
                                   ind_plot,0.0,alat,x0,b_vectors,fname,inputpath)

                Om_k = Om_kps = None
                    
                if ac_cond_spin:
                    ene_ac,sigxy = do_spin_Hall_conductivity(E_k,jksp,pksp,temp,0,npool,
                                                             ipol,jpol,shift,deltakp,deltakp2,smearing)
    
                omega = alat**3 * np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))
    
                if rank == 0:
                    shc *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/omega
                    f=open(os.path.join(inputpath,'shcEf_'+str(LL[spol])+'_'+str(LL[ipol])+str(LL[jpol])+'.dat'),'w')
                    for n in range(ene.size):
                        f.write('%.5f %9.5e \n' %(ene[n],shc[n]))
                    f.close()
    
                    if  ac_cond_spin:
                        sigxy *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/omega
                        f=open(os.path.join(inputpath,'SCDi_'+str(LL[spol])+'_'+str(LL[ipol])+str(LL[jpol])+'.dat'),'w')
                        for n in range(ene.size):
                            f.write('%.5f %9.5e \n' %(ene_ac[n],np.imag(ene_ac[n]*sigxy[n]/105.4571)))  #convert energy in freq (1/hbar in cgs units)
                        f.close()
                        f=open(os.path.join(inputpath,'SCDr_'+str(LL[spol])+'_'+str(LL[ipol])+str(LL[jpol])+'.dat'),'w')
                        for n in range(ene.size):
                            f.write('%.5f %9.5e \n' %(ene_ac[n],np.real(sigxy[n])))
                        f.close()

            comm.Barrier()
            if rank == 0:
                print('spin Hall module in              %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
                reset=time.time()
    
        dHksp = None
    except Exception as e:
        print('Rank %d: Exception in Spin Hall Module'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    
    v_k = None

    try:
        #----------------------
        # Compute Berry curvature and AHC
        #----------------------
        if Berry:
            if dftSO == False: sys.exit('full relativistic calculation with SO needed')
 
            for n in range(a_tensor.shape[0]):
                ipol = a_tensor[n][0]
                jpol = a_tensor[n][1]
                ene,ahc,Om_k = do_Berry_curvature(E_k,pksp,nk1,nk2,nk3,npool,ipol,jpol,
                                                  eminAH,emaxAH,fermi_dw,fermi_up,deltakp,smearing)
    
                if writedata:
                    if rank == 0: 
                        Om_kps = np.zeros((nk1,nk2,nk3,2),dtype=float)
                        x0 = np.zeros(3,dtype=float)
                        ind_plot = np.zeros(2)
                        Om_kps[:,:,:,0] = Om_k
                        Om_kps[:,:,:,1] = Om_k
                        fname = 'Berry_'+str(LL[ipol])+str(LL[jpol])+'.bxsf'
                        write2bxsf(fermi_dw,fermi_up,Om_kps,nk1,nk2,nk3,2,ind_plot,
                                   0.0,alat,x0,b_vectors,fname,inputpath)
    
                        np.savez(os.path.join(inputpath,'Berry_'+str(LL[ipol])+str(LL[jpol])+'.npz'),kq=kq,Om_k=Om_k[:,:,:])

                Om_k = Om_kps = None

                if ac_cond_Berry:
                    ene_ac,sigxy = do_Berry_conductivity(E_k,pksp,temp,0,npool,
                                                         ipol,jpol,shift,deltakp,deltakp2,smearing)

                omega = alat**3 * np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))
                if rank == 0:
                    ahc *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/omega
                    f=open(os.path.join(inputpath,'ahcEf_'+str(LL[ipol])+str(LL[jpol])+'.dat'),'w')
                    for n in range(ene.size):
                        f.write('%.5f %9.5e \n' %(ene[n],ahc[n]))
                    f.close()
    
                    if ac_cond_Berry:
                        sigxy *= 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/H_OVER_TPI/omega
                        f=open(os.path.join(inputpath,'MCDi_'+str(LL[ipol])+str(LL[jpol])+'.dat'),'w')
                        for n in range(ene.size):
                            f.write('%.5f %9.5e \n' %(ene_ac[n],np.imag(ene_ac[n]*sigxy[n]/105.4571)))  #convert energy in freq (1/hbar in cgs units)
                        f.close()
                        f=open(os.path.join(inputpath,'MCDr_'+str(LL[ipol])+str(LL[jpol])+'.dat'),'w')
                        for n in range(ene.size):
                            f.write('%.5f %9.5e \n' %(ene_ac[n],np.real(sigxy[n])))
                        f.close()
    
            comm.Barrier()
            if rank == 0:
                print('Berry module in                  %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
                reset=time.time()
    except Exception as e:
        print('Rank %d: Exception in Berry Module'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    


        #----------------------
        # Compute transport quantities (conductivity, Seebeck and thermal electrical conductivity)
        #----------------------
    if Boltzmann:

        if nspin==2:
            spin_mult=1.0
        else:
            if dftSO:
                spin_mult=1.0
            else:
                spin_mult=2.0


        temps = np.arange(tmin,tmax+1.e-10,tstep)

        # in cubic bohr radii 
        omega = alat**3 * np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))


        #restrict states that are within energy range of interest +- 1eV

        E_k_mask = np.where(np.logical_and(E_k[:,:bnd,:]>=(eminBT-1.0),E_k[:,:bnd,:]<=(emaxBT+1.0)))
        E_k_range = np.ascontiguousarray(E_k[E_k_mask[0],E_k_mask[1],E_k_mask[2]])
        velkp_range = np.swapaxes(velkp,1,0)

        velkp_range = np.ascontiguousarray(velkp_range[:,E_k_mask[0],E_k_mask[1],E_k_mask[2]])

        if carrier_conc:
            M_ij_range = np.ascontiguousarray(M_ij[:,E_k_mask[0],E_k_mask[1],E_k_mask[2]])    

        ispin=0
    # for ispin in range(nspin):
    #     E_k_mask = np.where(np.logical_and(E_k[:,:,ispin]>=(eminBT-1.0),E_k[:,:,ispin]<=(emaxBT+1.0)))
    #     E_k_range = np.ascontiguousarray(E_k[E_k_mask[0],E_k_mask[1],ispin])
    #     velkp_range = np.swapaxes(velkp,1,0)
    #     velkp_range = np.ascontiguousarray(velkp_range[:,E_k_mask[0],E_k_mask[1],ispin])

    #     if carrier_conc:
    #         M_ij_range = np.ascontiguousarray(M_ij[:,E_k_mask[0],E_k_mask[1],ispin])    




        sigmadk_str=""
        sigma_str=""
        kappa_str=""
        seebeck_str=""
        PF_str=""
        Hall_str=""

        for t in range(temps.shape[0]):
            temp=temps[t]/11604.52500617
            comm.Barrier()


            if smearing != None:
                ene,L0 = do_Boltz_tensors(E_k,velkp,kq_wght,temp,ispin,deltakp,smearing,t_tensor,eminBT,emaxBT,ne)

                #----------------------
                # Conductivity (in units of 1.e21/Ohm/m/s)
                #----------------------
                if rank==0:
                    L0 *= spin_mult    
                    L0 *= 1.0/omega
                    L0 *= 6.9884 # convert in units of 10*21 siemens m^-1 s^-1

                    sigma = L0*1.0e21 # convert in units of siemens m^-1 s^-1

                    if rank == 0:
                        for n in range(ene.size):
                            sigmadk_str+='%8.2f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e \n' \
                                %(temps[t],ene[n],sigma[0,0,n],sigma[1,1,n],sigma[2,2,n],sigma[0,1,n],sigma[0,2,n],sigma[1,2,n])
                    L0 = None


            ene,L0,L1,L2 = do_Boltz_tensors(E_k_range,velkp_range,kq_wght,temp,ispin,0.1,None,t_tensor,eminBT,emaxBT,ne)  


            if rank == 0:
                #----------------------
                # Conductivity (in units of /Ohm/m/s)
                #----------------------

                L0 *= spin_mult    
                L0 *= 1.0/omega
                L0 *= 6.9884 # convert in units of 10^21 siemens m^-1 s^-1

                sigma = L0*1.0e21 # convert in units of siemens m^-1 s^-1

                for n in range(ene.size):
                    sigma_str+='%8.2f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e \n' \
                        %(temps[t],ene[n],sigma[0,0,n],sigma[1,1,n],sigma[2,2,n],sigma[0,1,n],sigma[0,2,n],sigma[1,2,n])


                #----------------------
                # Seebeck (in units of V/K)
                #----------------------

                S = np.zeros((3,3,ene.size),dtype=float)

                L1 *= spin_mult
                L1 *= 1.0/omega
                L1 *= 6.9884 # convert in units of 10^21 Amperes m^-1 s^-1
                L1 *= 1.0/(temp*11604.52500617)

                inv_L0=np.zeros_like(L0)
                for n in range(ene.size):
                    try:
                        inv_L0[:,:,n] = LAN.inv(L0[:,:,n])
                    except:
                        inv_L0[:,:,n]= 0.0


                for n in range(ene.size):
                        S[:,:,n] = -1.0*np.dot(inv_L0[:,:,n],L1[:,:,n])


                for n in range(ene.size):
                    seebeck_str+='%8.2f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e \n' \
                        %(temps[t],ene[n],S[0,0,n],S[1,1,n],S[2,2,n],S[0,1,n],S[0,2,n],S[1,2,n])

                #----------------------
                # Electron thermal conductivity ((in units of W/m/K/s)
                #----------------------
                kappa = np.zeros((3,3,ene.size),dtype=float)

                L2 *= spin_mult
                L2 *= 1.0/omega
                L2 *= 6.9884e15 # convert in units of kg m s^-4
                L2 *= 1.0/(temp*11604.52500617) # temp in kelvin


                for n in range(ene.size):
                    try:
                        kappa[:,:,n] = (L2[:,:,n] - temp*11604.52500617*L1[:,:,n]*inv_L0[:,:,n]*L1[:,:,n])*1.e6
                    except:
                        kappa[:,:,n] = 0.0

                for n in range(ene.size):
                    kappa_str+='%8.2f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e \n' \
                        %(temps[t],ene[n],kappa[0,0,n],kappa[1,1,n],kappa[2,2,n],kappa[0,1,n],kappa[0,2,n],kappa[1,2,n])


                PF = np.zeros((3,3,ene.size),dtype=float)

                for n in range(ene.size):
                    PF[:,:,n] = np.dot(np.dot(S[:,:,n],L0[:,:,n]),S[:,:,n])*1.e21
                for n in range(ene.size):
                    PF_str+='%8.2f % .5f % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e % 9.5e \n' \
                        %(temps[t],ene[n],PF[0,0,n],PF[1,1,n],PF[2,2,n],PF[0,1,n],PF[0,2,n],PF[1,2,n])


            if carrier_conc:
                ene,sig_ijk = do_Hall_tensors(E_k_range,velkp_range,M_ij_range,kq_wght,temp,
                                          ispin,deltakp,smearing,t_tensor,eminBT,emaxBT,ne)


                if rank==0:
                    R_ijk = np.zeros_like(sig_ijk)

                    #return inverse L0 to base units
                    inv_L0 *= 1.0/omega
                    inv_L0 *= 6.9884 

                    #scale by cell size 

                    #multiply by spin multiplier
                    sig_ijk *= spin_mult


                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                for a in range(3):
                                    for b in range(3):
                                        R_ijk[i,j,k,:] += -inv_L0[a,j,:]*sig_ijk[a,b,k,:]*inv_L0[i,b,:]




                    for n in range(ene.size):
                        pcp = 3.0/(R_ijk[1,2,0,n]+R_ijk[2,0,1,n]+R_ijk[0,1,2,n])
                        pcpm = pcp/(omega*(5.29177249e-9**3))

                        Hall_str+='%8.2f % .5f % 9.5e % 9.5e \n' \
                            %(temps[t],ene[n],pcp,pcpm)

        if rank==0:
            if smearing!=None:
                f=open(os.path.join(inputpath,'sigmadk_'+str(ispin)+'.dat'),'w')
                f.write(sigmadk_str)
                f.close()
            f=open(os.path.join(inputpath,'sigma_'+str(ispin)+'.dat'),'w')
            f.write(sigma_str)
            f.close()
            f=open(os.path.join(inputpath,'Seebeck_'+str(ispin)+'.dat'),'w')
            f.write(seebeck_str)
            f.close()
            f=open(os.path.join(inputpath,'kappa_'+str(ispin)+'.dat'),'w')
            f.write(kappa_str)
            f.close()    
            f=open(os.path.join(inputpath,'PF_'+str(ispin)+'.dat'),'w')
            f.write(PF_str)
            f.close()    
            if carrier_conc:
                f=open(os.path.join(inputpath,'carrier_conc_'+str(ispin)+'.dat'),'w')
                f.write(Hall_str)
                f.close()    

        velkp = None
        L0 = None
        L1 = None
        L2 = None
        S = None
        kappa = None

        comm.Barrier()
        if rank ==0:
            print('transport in                     %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))

            reset=time.time()

    comm.Barrier()

    try:
        #----------------------
        # Compute dielectric tensor (Re and Im epsilon)
        #----------------------
        if epsilon:
            #from do_epsilon_d2 import *
    
            omega = alat**3 * np.dot(a_vectors[0,:],np.cross(a_vectors[1,:],a_vectors[2,:]))
    
            for n in range(d_tensor.shape[0]):
                ipol = d_tensor[n][0]
                jpol = d_tensor[n][1]
                for ispin in range(nspin):
                    ene, epsi, epsr, jdos = do_epsilon(E_k,pksp,kq_wght,omega,shift,delta,temp,ipol,jpol,ispin,metal,ne,epsmin,epsmax,deltakp,deltakp2,smearing,kramerskronig)
    
                    if rank == 0:
                        f=open(os.path.join(inputpath,'epsi_'+str(LL[ipol])+str(LL[jpol])+'_'+str(ispin)+'.dat'),'w')
                        for n in range(ene.size):
                            f.write('%.5f %9.5e \n' \
                                    %(ene[n],epsi[ipol,jpol,n]))
                        f.close()
                        f=open(os.path.join(inputpath,'epsr_'+str(LL[ipol])+str(LL[jpol])+'_'+str(ispin)+'.dat'),'w')
                        for n in range(ene.size):
                            f.write('%.5f %9.5e \n' \
                                    %(ene[n],epsr[ipol,jpol,n]))
                        f.close()
                        f=open(os.path.join(inputpath,'jdos_'+str(ispin)+'.dat'),'w')
                        for n in range(ene.size):
                            f.write('%.5f %9.5e \n' \
                                    %(ene[n],jdos[n]))
                        f.close()
    
            comm.Barrier()
            if rank ==0:
                print('epsilon in                       %5s sec ' %str('%.3f' %(time.time()-reset)).rjust(10))
    except Exception as e:
        print('Rank %d: Exception in Epsilon'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception
    
    try:
        # Timing
        if rank ==0:
            print('   ')
            print('Total CPU time =                 %5s sec ' %str('%.3f' %(time.time()-start)).rjust(10))
    except Exception as e:
        print('Rank %d: Exception in Total Time'%rank)
        traceback.print_exc()
        comm.Abort()
        raise Exception


    if verbose:

        mem = np.asarray(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if rank==0:
            print("Max total memory usage rank 0  :  %6.4f GB"%(mem/1024.0**2))




    if rank == 0:
        if 'Hksp' in outDict:
            outDict['Hksp'] = Hksp
        if 'v_k' in outDict:
            outDict['v_k'] = v_k
        if 'kq' in outDict:
            outDict['kq'] = kq
        if 'nk1' in outDict:
            outDict['nk1'] = nk1
        if 'nk2' in outDict:
            outDict['nk2'] = nk2
        if 'nk3' in outDict:
            outDict['nk3'] = nk3
        return outDict
    else:
        return
#def main():
#    datainput = sys.argv[1]
#    print('using the main program')
#    paoflow(datainput)
#
#if __name__== "__main__":
#    main()

