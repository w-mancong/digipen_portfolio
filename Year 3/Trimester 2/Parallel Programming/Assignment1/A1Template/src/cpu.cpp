/*
* Copyright 2022 Digipen.  All rights reserved.
*
* Please refer to the end user license associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms 
* is strictly prohibited.
*/
#include "heat.h"
#include <stdio.h>
extern "C" void initPoints(
	float *pointIn,
	float *pointOut,
	uint nRowPoints
)
{
	for (uint i = 0; i < nRowPoints; ++i)
	{
		for (uint j = 0; j < nRowPoints; ++j)
		{
			uint idx = i * nRowPoints + j;
			// (0, 10) to (0, 30) inclusive
			if (0 == i && 10 <= j && 30 >= j)
				*(pointIn + idx) = *(pointOut + idx) = 65.56f;
			// All the edges
			else if (0 == i || nRowPoints - 1 == i || 0 == j || nRowPoints - 1 == j)
				*(pointIn + idx) = *(pointOut + idx) = 26.67f;
			else // Interior points
				*(pointIn + idx) = *(pointOut + idx) = 0.0f;
		}
	}
}

extern "C" void heatDistrCPU(
	float *pointIn,
	float *pointOut,
	uint nRowPoints,
	uint nIter
)
{
	uint const TOTAL_COUNT = nRowPoints * nRowPoints;
	uint const INTERNAL_POINTS = nRowPoints - 1;
	for (uint k = 0; k < nIter; ++k)
	{
		for (uint i = 1; i < INTERNAL_POINTS; ++i)
		{
			for (uint j = 1; j < INTERNAL_POINTS; ++j)
			{
				uint const curr = (i * nRowPoints) + j,
					left		= ((i - 1) * nRowPoints) + j,
					right		= ((i + 1) * nRowPoints) + j,
					up			= (i * nRowPoints) + j + 1, 
					down		= (i * nRowPoints) + j - 1;

				*(pointOut + curr) =
					(
						*(pointIn + left) + 
						*(pointIn + right) + 
						*(pointIn + up) + 
						*(pointIn + down)
					) / 4;
			}
		}

		for (uint it = 0; it < TOTAL_COUNT; ++it)
			*(pointIn + it) = *(pointOut + it);
	}
}
