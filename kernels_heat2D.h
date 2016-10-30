#pragma once
#include <curand_kernel.h>

//generate uniform ramdom numbers
__device__ float generate( 

	curandState * globalState,
	int idx );

//setup the kernel 
__global__ void setup_kernel( 

	curandState * state, 
	unsigned long seed );

// initialize array with random numbers
__global__ void InitializeTemperature( 

	float * const ioArray, 
	curandState * globalState, 
	const int inArrSize );

// Run the heat transfer simulation
__global__ void HeatConduction2D( 

	const int inWidth,
	const int inHeight, 
	const float * const inTemp, 
	float * const ouTemp );
