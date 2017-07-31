
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
import sys, time

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as skfft

# Perform an inverse FFT on 'axes' of 'Hk'
# Restriction: len(Hk) >= len(axes)
# Restriction: 'axes' must be a list of unique, monotonically increasing integers
def cuda_ifftn ( Hk, axes=[0,1,2] ):
    return cuda_efftn(Hk, axes, False)

# Perform a FFT on 'axes' of 'Hr'
# Restriction: len(Hr) >= len(axes)
# Restriction: 'axes' must be a list of unique, monotonically increasing integers
def cuda_fftn ( Hr, axes=[0,1,2] ):
    return cuda_efftn(Hr, axes, True)

# Handle array shaping and FFT planning
def cuda_efftn ( H, axes, forward ):

    hShape = H.shape
    hDim = len(hShape)
    fftDim = len(axes)

    # Reshape 'axes' to be the array's end dimensions and ensure contiguity
    H = np.ascontiguousarray(np.moveaxis(H, axes, np.arange(hDim-fftDim, hDim, 1)))
    newShape = H.shape

    # Calculate number of batches
    batchSize = 1
    for i in range(hDim-fftDim):
        batchSize *= H.shape[i]

    # Reshape to accomodate batching
    batchShape = [None for _ in np.arange(fftDim+1)]
    batchShape[0] = batchSize
    for i in np.arange(0, fftDim, 1):
        batchShape[i+1] = newShape[hDim-(fftDim-i)]

    H = np.reshape(H, batchShape)

    # Pass array to the GPU and perform iFFT on each batch
    H_gpu = gpuarray.to_gpu(H)
    plan = skfft.Plan(H_gpu.shape[1:fftDim+1], H.dtype, H.dtype, H_gpu.shape[0])

    if forward:
        skfft.fft(H_gpu, H_gpu, plan)
    else:
        skfft.ifft(H_gpu, H_gpu, plan, True)

    # Reshape to original dimensions
    H = np.reshape(H_gpu.get(), newShape)
    H = np.moveaxis(H, np.arange(hDim-fftDim, hDim, 1), axes)

    return H
