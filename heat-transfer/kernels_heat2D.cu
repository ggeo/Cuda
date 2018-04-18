#include <curand_kernel.h>


__device__ float generate( 

	curandState * globalState,
    int idx )
{
	idx = threadIdx.x + blockIdx.x * blockDim.x;
	curandState localState = globalState[ idx ];
	float RANDOM = curand_uniform( &localState );
	globalState[ idx ] = localState;

	return RANDOM;
}

__global__ void setup_kernel( 

	curandState * state, 
	unsigned long seed )
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init( seed, id, 0, &state[ id ] );

	return;
}

__global__ void InitializeTemperature( 

	float * const ioArray, 
	curandState * globalState, 
	const int inArrSize )
{
	// generate random numbers
	for ( int i = 0; i < inArrSize; i++ )
	{
		float k = generate( globalState, i );
		ioArray[ i ] = k;
	}


	return;
}


/* ========================================================================== */
/*   HeatConduction                                                           */
/* -------------------------------------------------------------------------- */
/*!
 * @function HeatConduction2D
 *
 * @abstract function to calculate the 2D heat conduction in a body
 *
 * @ Initial conditions: 
 		
 	T(0,y,t) = T(Lx,y,t) = 0
 	T(x,0,t) = T(x,Ly,t) = 0
 	T(x,y,0) = initial temperature
 	0 <= x <= Lx
 	0 <= y <= Ly
 	0 <= t <= T	
 	
	We are assuming a square body and we divide it to small squares each of 
	them having a temperature.
	The temperature flows from warmer to colder square.
	Temperature can flow to the neighboor squares (left,right,top,bottom)
	
	We are using the appropriate offset in order to move in the above places.
	We are applying the appropriate boundary conditions when trying to move
	to neighbor places.

 * @param inWidth [ input ] The width of the body (cm)
 *
 * @param inHeight [ input ] The heoght of the body (cm)
 *
 * @param inTemp [ input ] The initial temperature of body
 * 
 * @param ouTemp [ output ] The temperature of body after solving the system
 */
/* ========================================================================== */
__global__ void HeatConduction2D( 

	const int inWidth,
	const int inHeight, 
	const float * const inTemp, 
	float * const ouTemp )
{

	int rowIdx = threadIdx.y + blockIdx.y * blockDim.y;
	int colIdx = threadIdx.x + blockIdx.x * blockIdx.x;
	int offset = rowIdx * inWidth + colIdx;

	if ( rowIdx >= inHeight || colIdx >= inWidth ) return;

	// new offsets 
	int left = offset - 1;
	int right = offset + 1;
	int top = offset + inWidth;
	int bottom = offset - inWidth;

	//boundary conditions
	if ( 0 == colIdx ) left += inWidth;
	if ( inWidth - 1 == colIdx ) right -= inWidth;
	if ( 0 == rowIdx ) bottom += inWidth * inHeight;
	if ( inHeight - 1 == rowIdx ) top -= inWidth * inHeight;

	ouTemp[ offset ] = inTemp[ offset ] + (1.f/4.f) * ( inTemp[ left ] + inTemp[ right ] + inTemp[ top ] + inTemp[ bottom ] - 4 * inTemp[ offset ] );

}
