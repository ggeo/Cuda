#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include "kernels_heat2D.h"

using namespace std;

//----function to check for errors-------------------------------------------------
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert( cudaError_t code, char * file, int line, bool abort = true )
{
	if ( cudaSuccess != code )
	{
		fprintf( stderr, "\nGPUassert: %s %s %d\n", cudaGetErrorString( code ), file, line );
		if ( abort )
			exit( code );
	}
}

int main()
{
	const int N = 256; // random numbers
	const int width = 16;
	const int height = 16; 
	const int NbOfSteps = 4;
	const int NbOfThreadsX = 16;
	const int NbOfThreadsY = 16;
	const int NbOfBlocksX = ceil(height / NbOfThreadsX);
	const int NbOfBlocksY = ceil(width / NbOfThreadsY);


	// allocate host memory
	float *ouTemp = new float[ width * height ];

	//allocate device memory
	float *devinTemp, *devouTemp;
	gpuErrchk( cudaMalloc( (void **) &devinTemp, width * height * sizeof(*devinTemp) ) );
	gpuErrchk( cudaMalloc( (void **) &devouTemp, width * height * sizeof(*devouTemp) ) );

	/* ==== Generate random numbers for initializing temperature ==== */
	curandState * devStates;
    	gpuErrchk( cudaMalloc( &devStates, N * sizeof(curandState) ) );

    	// setup seeds
    	setup_kernel<<<1,N>>>( devStates, static_cast<unsigned>(time(NULL)) );
    	gpuErrchk( cudaPeekAtLastError() );
    	gpuErrchk( cudaDeviceSynchronize() );

    	// Initialize temperature
    	InitializeTemperature<<<1,256>>>( devinTemp, devStates, width * height );
    	gpuErrchk( cudaPeekAtLastError() );
    	gpuErrchk( cudaDeviceSynchronize() );
    	/* ============================================================== */

	// specify threads and blocks
	dim3 theThreadsPerBlock( NbOfThreadsX, NbOfThreadsX );
	dim3 theBlocksPerGrid( NbOfBlocksX, NbOfBlocksY );

   	 /*  Run the kernel by number of time steps
    	     Because we are executing the default stream, kernel
       	     calls are executed serially.
    	     Supply the output of 1st kernel call (the output temperatures)
    	     as input to the second in order for the temperatures to be
    	     updated in the next step
    	*/
    	for ( int k = 0; k < NbOfSteps; k++ )
    	{

		HeatConduction2D<<< theBlocksPerGrid, theThreadsPerBlock >>>( 

			width,
			height,
			devinTemp,
			devouTemp );

	#ifdef DEBUG
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
	#endif

		HeatConduction2D<<< theBlocksPerGrid, theThreadsPerBlock >>>( 

			width,
			height,
			devouTemp,
			devinTemp );

	#ifdef DEBUG
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
	#endif
	}

	// copy output back to CPU
	gpuErrchk( cudaMemcpy( ouTemp, devouTemp, width * height * sizeof(*devouTemp), cudaMemcpyDeviceToHost ) );

	ofstream theFile ("heat2d_cuda.dat");

	// print to file
	for ( int i = 0; i < height; i++ )
	{
		for ( int j = 0; j < width; j++ )
		{
			theFile << ouTemp[ i * width + j ] << endl;
		}
	}

	theFile.close();

	//free memory
	delete [] ouTemp;
	gpuErrchk( cudaFree( devinTemp ) );
	gpuErrchk( cudaFree( devouTemp ) );

	return 0;

}
