#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <sys/time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <float.h>
#include <cmath>


/*
 *	This macro checks for API errors in the CUDA calls.
 */

#define gpuErrchk(ans) { gpuAssert( (ans), __FILE__, __LINE__ ); }

inline void
gpuAssert( cudaError_t code, const char * file, int line, bool abort = true )
{
	if ( cudaSuccess != code )
	{
		fprintf( stderr, "\nGPUassert: %s %s %d\n", cudaGetErrorString( code ), file, line );
		if ( abort )
			exit( code );
	}


	return;

} /* gpuAssert */

// Points struct
typedef struct
{
	float X,Y;
	int Value;

} Points;

/* ========================================================================== */
/*   VoronoiShared                                                                 */
/* -------------------------------------------------------------------------- */
/*!
 * @function    VoronoiShared
 *
 * @abstract
 *
 * @discussion  Calculates Voronoi cells
 *
 * @param	inNbOfSites [input] The number of the sites (seeds).
 * 								type: const size_t
 * 							
 * @param	inX [input] The x coordinates of the points
 *						Dimensions :  Nx , type: float
 * 
 * @param	inY [input] The y coordinates of the points
 *						Dimensions :  Ny , type: float		
 * 				  
 * @param   inV [input] The inV holds for applying a threshold/color
 * 						to the cell region
 *						Dimensions : inNbOfSites, type: int
 * 		    
 * @param	ouVoronoi [output] The output data (pixels)
 *          Dimensions :  The total number of threads in the grid
 *          ( theBlocksPerGridX * theBlocksPerGridY * theThreadsPerBlockX * theThreadsPerBlockY )
 *			type: float
 */
 /* ========================================================================== */

__global__ void VoronoiShared(

		size_t  const inNbOfSites,
		float * const inX,
		float * const inY,
		int   * const inV,
		int   * const ouVoronoi )
{

	int x = ( blockIdx.x * blockDim.x ) + threadIdx.x; // x coordinate of Pixel
	int y = ( blockIdx.y * blockDim.y ) + threadIdx.y; // y coordinate of Pixel
	int theGlobalIdx = x + y * ( blockDim.x * gridDim.x );

	extern __shared__ Points points[ ];
	float distX , distY;
	float theTempDistance ,theDistance = FLT_MAX;
	int theThreshold;

	//loop through all points calculating distance
	if ( x < blockDim.x * gridDim.x && y < blockDim.y * gridDim.y ) //ensure that only the threads we need execute the code
	{
		for ( int i = 0; i < inNbOfSites; i++ )
		{
			//load data to shared memory
			points[ gridDim.x * threadIdx.y + threadIdx.x ].X 	   = inX[ i ];
			points[ gridDim.x * threadIdx.y + threadIdx.x ].Y 	   = inY[ i ];
			points[ gridDim.x * threadIdx.y + threadIdx.x ].Value  = inV[ i ];

			__syncthreads();

			//Calculate distances for all the points
			for ( int j = 0; j < inNbOfSites; j++ )
			{

				distX = points[ j ].X - x;
				distY = points[ j ].Y - y;
				distX *= distX;
				distY *= distY;
				theTempDistance = distX + distY;

				//if this Point is closer , assign proper threshold
				if ( theTempDistance < theDistance )
				{
					theDistance = theTempDistance;
					theThreshold = points[ j ].Value;
				}
			}

			__syncthreads();
		}

	}
	//write result back to global memory
	*( ouVoronoi + theGlobalIdx ) = theThreshold;

}
	
int main()
{
	const size_t Width = 256 , Height = 256;
	const size_t Nx = 128 , Ny = 128;
	const size_t NbOfSites = 100; //should be <= Nx and Ny
	const size_t ThreadsPerBlockX = 16 , ThreadsPerBlockY = 16 ,BlocksPerGridX = Width / 16 , BlocksPerGridY = Height / 16;
	const size_t TotalNbOfPixels = ( Width * Height );
	
	// Allocate host memory
	float * X = (float*) malloc( Nx * sizeof (*X) );
	assert( NULL != X );
	float * Y = (float*) malloc( Ny * sizeof (*Y) );
	assert( NULL != Y );
	int * V = (int*) malloc( NbOfSites * sizeof (*V) );
	assert( NULL != V );
	int * VoronoiDiagram = (int*) malloc ( TotalNbOfPixels * sizeof(*VoronoiDiagram) );
	assert( NULL != VoronoiDiagram );
	
	float * devX , * devY;
	int * devVoronoiDiagram , * devV;
	// Allocate device memory
	gpuErrchk( cudaMalloc( (void**) &devX, Nx * sizeof(*devX) ) );
	gpuErrchk( cudaMalloc( (void**) &devY, Ny * sizeof(*devY) ) );
	gpuErrchk( cudaMalloc( (void**) &devV, NbOfSites * sizeof(*devV) ) );
	gpuErrchk( cudaMalloc( (void**) &devVoronoiDiagram,  TotalNbOfPixels * sizeof(*devVoronoiDiagram) ) );
	
	// Create random coordinates
	srand((unsigned int)time(NULL));
	for ( int i = 0; i < Nx; i++ )	X[ i ] = ( ( (float) rand() / (float) ( RAND_MAX ) ) * Width );
	for ( int i = 0; i < Ny; i++ )  Y[ i ] = ( ( (float) rand() / (float) ( RAND_MAX ) ) * Height );
	
	for ( int i = 0; i < NbOfSites; i++ )	V[ i ] = i;
	
	// Define grid dimensions
	dim3 BlocksDim ( BlocksPerGridX , BlocksPerGridY );
	dim3 ThreadsPerBlock ( ThreadsPerBlockX , ThreadsPerBlockY );
	
	gpuErrchk( cudaMemcpy( devV , V , NbOfSites * sizeof( *V ), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy( devX , X , Nx * sizeof( *X ), cudaMemcpyHostToDevice ) );
	gpuErrchk( cudaMemcpy( devY , Y , Ny * sizeof( *Y ), cudaMemcpyHostToDevice ) );
		
  	cudaEvent_t CurrentEventPre,
  	CurrentEventPost;
  	float CurrentPostPreTimeMS;
  
  	gpuErrchk( cudaEventCreate( &CurrentEventPre ) );
  	gpuErrchk( cudaEventCreate( &CurrentEventPost ) );
  
  	gpuErrchk( cudaEventRecord( CurrentEventPre ) );

  	size_t DynamicSharedSize = ThreadsPerBlockX + ThreadsPerBlockY * BlocksPerGridX;
  	VoronoiShared<<< BlocksDim,
  					 ThreadsPerBlock,
  					 DynamicSharedSize * sizeof(Points)>>>
  					 ( NbOfSites, 
						devX,
						devY,
						devV,
						devVoronoiDiagram );
  
  	gpuErrchk( cudaPeekAtLastError() );
  	gpuErrchk( cudaDeviceSynchronize() );
  
  	gpuErrchk( cudaEventRecord( CurrentEventPost ) );
  	gpuErrchk( cudaEventSynchronize( CurrentEventPost ) );
  	gpuErrchk( cudaEventElapsedTime( &CurrentPostPreTimeMS, CurrentEventPre, CurrentEventPost ) );
  	printf( "\nGPU time for calling Voronoi: %f ms\n", CurrentPostPreTimeMS );
  
  	gpuErrchk( cudaMemcpy( VoronoiDiagram,
  						   devVoronoiDiagram , 
  						   TotalNbOfPixels * sizeof(*devVoronoiDiagram), cudaMemcpyDeviceToHost ) );
  
  	{
		FILE * theFile;		
	  	theFile = fopen( "VoronoiShared", "wb" );
	  	assert( NULL != theFile );
	  	assert( TotalNbOfPixels == fwrite( VoronoiDiagram , sizeof(*devVoronoiDiagram), TotalNbOfPixels , theFile ) );
	  	fclose( theFile );
  	}
  
  	//free memory
  	gpuErrchk( cudaFree( devX ) );
  	gpuErrchk( cudaFree( devY ) );
  	gpuErrchk( cudaFree( devV ) );
  	gpuErrchk( cudaFree( devVoronoiDiagram ) );
  
  	free( X );
  	free( Y );
  	free( V );
  	free( VoronoiDiagram );
  
  	return 0;
}
						