/*
* Copyright 2024 Digipen.  All rights reserved.
*
* Please refer to the end user license associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms
* is strictly prohibited.
*
*/
#include <helper_cuda.h>
#include "helper.h"

/**************
	The steps for the algorithm:
	1) Declare shared memory array for N elements with the size of TILE_WIDTH_RATIO_K × TILE_WIDTH_N
	2) Declare output array variable for P elements with the size of TILE_WIDTH_N and initialize the output array variable
	3) Loop over the input tiles (the number of iterations = (k - 1) / TILE_WIDTH_RATIO_K + 1, where k is the number of the columns of matrix M.)
	   a) Load the tile of N (size = TILE_WIDTH_RATIO_K × TILE_WIDTH_N) into shared memory.
	   Note: one block has TILE_WIDTH_M threads, each loading one N element into shared memory.
	   b) Loop over elements inside the tile of N (the number of iteration = TILE_WIDTH_RATIO_K).
	      i. Load tile of matrix M into register (i.e. each thread load one M element into the local variable in this iteration)
		 ii. Loop over and update the output elements in the output array variable for P elements assigned to this thread.
			 Note: output array variable are local variables. They accumulate the partial results. In this innerloop, the number of iteration is TILE_WIDTH_N
	4) Store the output array variable to P elements (each thread stores TILE_WIDTH_N P elements and one block outputs TILE_WIDTH_N × TILE_WIDTH_M P elements).
****************************/

//P and M column-major, N row-major
__global__ 
void matrixMultiply(FLOAT_TYPE *P,       //<! [out] and mxn matrix
					FLOAT_TYPE const *M, //<! [in] an mxk matrix
					FLOAT_TYPE const *N, //<! [in] an kxn matrix
					int const m, int const n, int const k)
{
	// Shared memory for tiling input N array
	__shared__ FLOAT_TYPE N_s[TILE_WIDTH_RATIO_K][TILE_WIDTH_N]; // 1)
	FLOAT_TYPE P_local[TILE_WIDTH_N] = { 0 };	// 2) 
	for (int i = 0; i < TILE_WIDTH_N; ++i)
		P_local[i] = 0.0f;

	int const bx = blockIdx.x;
	int const by = blockIdx.y;
	int const tx = threadIdx.x;

	// Index to access the shared memory, using threadIdx.x
	int const Ns_col = tx % TILE_WIDTH_N;
	int const Ns_row = tx / TILE_WIDTH_N;

	// Since this is independent from each steps, can just precalculate it here first
	int const Ng_col = Ns_col + (by * TILE_WIDTH_N); // Index to access global N's col
	int const Mg_row = tx + (bx * TILE_WIDTH_M);	 // Index to access global M's row

	// 3)
	int const K_IT = ((k - 1) / TILE_WIDTH_RATIO_K) + 1;
	for (int start_k = 0; start_k < K_IT; ++start_k)
	{
		// Index to access global N matrix
		int Ng_row = Ns_row + (start_k * TILE_WIDTH_RATIO_K);

		// 3a) Each thread load from global N matrix into shared memory
		if (Ng_row < k && Ng_col < n)
			N_s[Ns_row][Ns_col] = N[Ng_row * n + Ng_col];
		__syncthreads();

		// 3b)
		for (int it = 0; it < TILE_WIDTH_RATIO_K; ++it)
		{
			// Index to access global M matrix
			int Mg_col = it + (start_k * TILE_WIDTH_RATIO_K);

			// Check if Mg_col and Mg_row is within array's boundary, else set v to 0.0f
			FLOAT_TYPE const v = Mg_col < k && Mg_row < m ? M[Mg_row + m * Mg_col] : 0.0f;

			for (int n_offset = 0; n_offset < TILE_WIDTH_N; ++n_offset)
				P_local[n_offset] += v * N_s[it][n_offset];
		}
		__syncthreads(); // calling syncthreads here so that slower threads can still access the shared memory
	}

	// 4)
	for (int i = 0; i < TILE_WIDTH_N; ++i)
	{
		int P_col = (by * TILE_WIDTH_N) + i;
		int P_row = tx + (bx * TILE_WIDTH_M);

		if (P_col < n && P_row < m)
			P[P_row + m * P_col] = P_local[i];
	}
}

void matrixMultiplyGPU(FLOAT_TYPE* P,
	FLOAT_TYPE* M,
	FLOAT_TYPE* N,
	int numMRows,
	int numNColumns,
	int numMColumns)
{
	//@@ Initialize the grid and block dimensions here

	dim3 dimGrid((numMRows - 1) / TILE_WIDTH_M + 1, (numNColumns - 1) / TILE_WIDTH_N + 1);
	dim3 dimBlock(TILE_WIDTH_M, 1);

	matrixMultiply<<<dimGrid, dimBlock>>>(P, M, N, numMRows, numNColumns, numMColumns);

	getLastCudaError("matrixMultiply failed\n");
	cudaDeviceSynchronize();
}

// To print out the M_reg value
//if (tx == 0)
//{
//	printf("Block (%d, %d), M[%d * %d + %d] = %f\n", bx, by, Mg_row, m, Mg_col, v);
//}

// To print out the values inside P_reg
//for (int i = 0; i < TILE_WIDTH_N; ++i) 
//{
//	if (P_local[i] > 0.0f || P_local[i] < 0.0f)
//	{
//		printf("Block (%d, %d), Thread (%d, %d), P_local[%d] = %f\n",
//			bx, by, tx, threadIdx.y, i, P_local[i]);
//	}
//}
//__syncthreads();

// Check if N_s value is correct
//if (threadIdx.x == 0) {
//	for (int x = 0; x < TILE_WIDTH_RATIO_K; ++x) {
//		for (int y = 0; y < TILE_WIDTH_N; ++y) {
//			printf("Block (%d, %d), N_s[%d][%d] = %f\n", blockIdx.x, blockIdx.y, x, y, N_s[x][y]);
//		}
//	}
//}
//__syncthreads();