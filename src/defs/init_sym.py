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
import spglib as spg

# initialize symmetry data

def init_sym(data_controller):

	arrays,attr = data_controller.data_dicts()
	
	numbers = arrays['numbers']
	cell = (arrays['a_vectors'],arrays['tau']/attr['alat'],numbers)
	mesh = [attr['nfft1'],attr['nfft2'],attr['nfft3']]
	mapping, grid = spg.get_ir_reciprocal_mesh(mesh, cell, is_shift=[0, 0, 0])
	_,irk,inv,irw = np.unique(mapping,return_index=True,return_inverse=True,return_counts=True)
	arrays['grid'] = grid/mesh
	arrays['irw'] = irw 
	arrays['irk'] = irk 