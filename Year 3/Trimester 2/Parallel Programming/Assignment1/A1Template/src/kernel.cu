/*
* Copyright 2022 Digipen.  All rights reserved.
*
* Please refer to the end user license associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms
* is strictly prohibited.
*/

#include <helper_cuda.h>
#include <cuda_runtime.h>
////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE 32
typedef unsigned int uint;
__global__ void heatDistrCalc(float* in, float* out, uint nRowPoints)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	// Check if x and y is within the grid boundaries
	if (nRowPoints <= x || nRowPoints <= y)
		return;

	// Calculate the index of the current point in the array
	uint idx = y * nRowPoints + x;
	uint interior = nRowPoints - 1;
	if (0 < x && interior > x && 0 < y && interior > y)
	{
		out[index] = (
						in[idx - 1] +
						in[idx + 1] +
						in[idx - nRowPoints] +
						in[idx + nRowPoints]
					 ) / 4.0f;
	}
}


__global__ void heatDistrUpdate(float* in, float* out, uint nRowPoints)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	// Check if x and y is within the grid boundaries
	if(nRowPoints <= x || nRowPoints <= y)
		return;

	uint idx = y * nRowPoints + x;
	uint interior = nRowPoints - 1;
	if(0 < x && interior > x && 0 < y && interior > y)
		in[idx] = out[idx];
}

extern "C" void heatDistrGPU(
	float* d_DataIn,
	float* d_DataOut,
	uint nRowPoints,
	uint nIter
)
{
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 gridSize(ceil(((float)nRowPoints) / BLOCK_SIZE), ceil(((float)nRowPoints) / BLOCK_SIZE), 1);

	//uint const TOTAL_COUNT = nRowPoints * nRowPoints;

	for (uint k = 0; k < nIter; ++k)
	{
		// Launch the heatDistrCalc kernel with the chosen grid and block dimensions
        heatDistrCalc<<<gridSize, blockSize>>>(d_DataIn, d_DataOut, nRowPoints);
		getLastCudaError("heatDistrCalc failed\n");
		// wait for all kernels to finish execution before continuing
		cudaDeviceSynchronize();

		//call heatDistrUpdate
		heatDistrUpdate<<<gridSize, blockSize>>>(d_DataIn, d_DataOut, nRowPoints);
		getLastCudaError("heatDistrUpdate failed\n");
		// wait for all kernels to finish execution before continuing
		cudaDeviceSynchronize();

		float* tmp = d_DataIn;
		d_DataIn = d_DataOut;
		d_DataOut = tmp;

		// Swap pointer for next iteration
		//for(uint i = 0; i < TOTAL_COUNT; ++i)
		//	*(d_DataOut + i) = *(d_DataIn + i);
	}
}
