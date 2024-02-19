/*
 * Copyright 2022 Digipen.  All rights reserved.
 *
 * Please refer to the end user license associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms
 * is strictly prohibited.
 *
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
/*
 * This sample implements Matrix Multiplication
 */

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h> // helper for shared that are common to CUDA Samples
#include "helper.h"

#include <stdint.h>

#define epsilon 1.0e-3

void printMatrix(FLOAT_TYPE *matrix, int numRows, int numCols)
{
	for (int i = 0; i < numRows; i++)
	{
		for (int j = 0; j < numCols; ++j)
		{
			printf("%f ", matrix[i * numCols + j]);
		}
		printf("\n");
	}
}

void correctness_test(int nRun, int numMRows, int numMCols, int numNCols)
{
    for (int i = 0; i < nRun; i++)
    {
        // Generate random matrices M and N
        FLOAT_TYPE* M = createData(numMRows, numMCols);
        FLOAT_TYPE* N = createData(numMCols, numNCols);

        // Print the matrix in the cpu
        //printf("M:\n");
        //printMatrix(M, numMRows, numMCols);
        //printf("N:\n");
        //printMatrix(N, numMCols, numNCols);

        // Allocate space for the result matrices
        FLOAT_TYPE* P_cpu = static_cast<FLOAT_TYPE*>(malloc(numMRows * numNCols * sizeof(FLOAT_TYPE)));
        FLOAT_TYPE* P_gpu = static_cast<FLOAT_TYPE*>(malloc(numMRows * numNCols * sizeof(FLOAT_TYPE)));

        // Perform matrix multiplication on the CPU
        matrixMultiplyCPU(P_cpu, M, N, numMRows, numMCols, numNCols);

        // Allocate device memory
        FLOAT_TYPE* d_M, * d_N, * d_P;
        cudaMalloc(reinterpret_cast<void**>(&d_M), numMRows * numMCols * sizeof(FLOAT_TYPE));
        cudaMalloc(reinterpret_cast<void**>(&d_N), numMCols * numNCols * sizeof(FLOAT_TYPE));
        cudaMalloc(reinterpret_cast<void**>(&d_P), numMRows * numNCols * sizeof(FLOAT_TYPE));

        // Convert M to column-major format
        FLOAT_TYPE* h_M_conv = static_cast<FLOAT_TYPE*>(malloc(numMRows * numMCols * sizeof(FLOAT_TYPE)));
        convertRowColumn(h_M_conv, M, numMRows, numMCols);

        // Copy matrices to the device
        checkCudaErrors(cudaMemcpy(d_M, h_M_conv, numMRows * numMCols * sizeof(FLOAT_TYPE), cudaMemcpyHostToDevice));
        getLastCudaError("cudaMemcpy to d_M failed\n");
        checkCudaErrors(cudaMemcpy(d_N, N, numMCols * numNCols * sizeof(FLOAT_TYPE), cudaMemcpyHostToDevice));
        getLastCudaError("cudaMemcpy to d_N failed\n");

        // Perform matrix multiplication on the GPU
        matrixMultiplyGPU(d_P, d_M, d_N, numMRows, numNCols, numMCols);

        // Copy the result back to host
        FLOAT_TYPE* h_P_2 = static_cast<FLOAT_TYPE*>(malloc(numMRows * numNCols * sizeof(FLOAT_TYPE)));
        checkCudaErrors(cudaMemcpy(h_P_2, d_P, numMRows * numNCols * sizeof(FLOAT_TYPE), cudaMemcpyDeviceToHost));
        getLastCudaError("cudaMemcpy to h_P_2 failed\n");

        // Convert P back to row-major format
        convertRowColumn(P_gpu, h_P_2, numNCols, numMRows);

        // Print the matrix in the cpu
        //printf("CPU result:\n");
        //printMatrix(P_cpu, numMRows, numNCols);

        //// Print the matrix in the gpu
        //printf("GPU result:\n");
        //printMatrix(P_gpu, numMRows, numNCols);

        // Compare CPU and GPU results
        bool ok = true;
        for (int j = 0; j < numMRows * numNCols; j++)
        {
            if (fabs(P_cpu[j] - P_gpu[j]) > epsilon)
            {
                ok = false;
                break;
            }
        }

        if (ok)
        {
            printf("Test passed!\n");
        }
        else
        {
        	printf("Test failed!\n");
		}

        // Free memory
        free(M);
        free(N);
        free(P_cpu);
        free(P_gpu);
        free(h_M_conv);
        free(h_P_2);
        cudaFree(d_M);
        cudaFree(d_N);
        cudaFree(d_P);
    }
}


void efficiency_test(int nRun, int numMRows, int numMCols, int numNCols)
{
    // lazy do
}

int main(int argc, char **argv)
{
	// int numMRows = 191;
	// int numMCols = 19;
	// int numNCols = 241;
	// int numNRows = numMCols;

	correctness_test(1, 101 - rand() % 10, 101 - rand() % 10, 101 - rand() % 10);
	correctness_test(1, 200 + rand() % 100, 200 + rand() % 100, 200 + rand() % 100);
	correctness_test(1, 500 + rand() % 500, 500 + rand() % 500, 500 + rand() % 500);
	correctness_test(1, 2000, 2000, 2000);

	// efficiency_test(10, 100, 100, 100);
	// efficiency_test(10, 500, 500, 500);
	// efficiency_test(10, 1000, 1000, 1000);

	return 0;
}
