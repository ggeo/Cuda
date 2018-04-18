#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>


int  main(int argc, char**argv)
{
    
	const int  rows = 3, cols = 2;

    //size in bytes
    const int ARRAY_BYTES = ( rows * cols ) * sizeof(int);
    
	float *A;
	A = (float *) malloc(ARRAY_BYTES);
	
	//initialize
	A[ 0 ] = 0;
	A[ 1 ] = 1;
	A[ 2 ] = 2;
	A[ 3 ] = 3;
	A[ 4 ] = 4;
	A[ 5 ] = 5;
			
	// print matrix
	printf("\nA matrix");
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j)
			printf("\nA = %f",A[ i + rows * j ]);
		printf("\n");
	}

	float *A_dev , *C_dev;
	cudaMalloc((void **) &A_dev, ARRAY_BYTES);
	cudaMalloc((void **) &C_dev, ARRAY_BYTES);
	
	cudaMemcpy(A_dev, A, ARRAY_BYTES, cudaMemcpyHostToDevice);

	float const alpha(1.0);
    float const beta(0.0);
    
	cublasHandle_t handle;

	cublasStatus_t status;

	status = cublasCreate(&handle);
	
	//use cublasSetPointerMode  HOST in order to be able to use alpha and beta in host ,else you must define them in device
	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST); 
	status = cublasSgeam(handle,CUBLAS_OP_T, CUBLAS_OP_N,  cols, rows , &alpha ,A_dev ,rows , &beta ,A_dev ,rows, C_dev , cols);

 	cudaMemcpy(A,C_dev, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	printf("\nA transposed ");
	for (int  i = 0; i < cols; ++i) {
		for (int j = 0; j < rows; ++j)
			printf("\nA = %f", A[ i + cols * j ]);
		printf("\n");
	}
  
	cudaFree(A_dev);
	cudaFree(C_dev);
	
	free(A);
	
    return 0;
}


