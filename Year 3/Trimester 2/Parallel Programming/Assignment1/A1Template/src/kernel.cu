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
////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE 32
typedef unsigned int uint;
__global__ void heatDistrCalc(float* in, float* out, uint nRowPoints)
{

}


__global__ void heatDistrUpdate(float* in, float* out, uint nRowPoints)
{

}

extern "C" void heatDistrGPU(
	float* d_DataIn,
	float* d_DataOut,
	uint nRowPoints,
	uint nIter
)
{
	dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 DimGrid2(ceil(((float)nRowPoints) / BLOCK_SIZE), ceil(((float)nRowPoints) / BLOCK_SIZE), 1);

	for (uint k = 0; k < nIter; k++) {
		//call heatDistrCalc
		getLastCudaError("heatDistrCalc failed\n");
		//synchronize
		//call heatDistrUpdate
		getLastCudaError("heatDistrUpdate failed\n");
		//synchronize
	}
}
