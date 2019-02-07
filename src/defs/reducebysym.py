#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016-2018 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
#
# Reference:
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#

import numpy as np

# Reduce array to the irreducible k-points in the BZ

def reducebysym(data_controller,arry):
	
	arrays,attr = data_controller.data_dicts()

	tmp = list(arry.shape)
	tmp[0] = arrays['irw'].shape[0]
	newshape = tuple(tmp)
		
	aux = np.zeros(newshape,dtype=arry.dtype)
	for n in range(arrays['irw'].shape[0]):
		aux[n] =  arry[arrays['irk'][n]]

	return(aux)
	
def reducebysym2(data_controller,arry):
	
	arrays,attr = data_controller.data_dicts()

	tmp = list(arry.shape)
	tmp[0] = arrays['irw'].shape[0]
	newshape = tuple(tmp)
		
	aux = np.zeros(newshape,dtype=arry.dtype)
	for n in range(arrays['irw'].shape[0]):
		nir = np.argwhere(arrays['mapping']==arrays['irk'][n])[:,0].shape[0]
		idx = np.argwhere(arrays['mapping']==arrays['irk'][n])[:,0]
		
		for i in range(nir):
			aux[n] +=  arry[idx[i]]/nir
			
	return(aux)
