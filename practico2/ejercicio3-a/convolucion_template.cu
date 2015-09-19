/*******************/
/** Ejercicio 3 a **/
/*******************/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <Windows.h>
#include "util.h"
#include <time.h>
//#include <sys\time.h>

#define CHUNK 256
#define SIZE_X 1048576
#define MASK_SIZE 21

void cudaCheck()
{
  cudaError_t cudaError;

  cudaError = cudaGetLastError();
  if(cudaError != cudaSuccess)
  {
    printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
  }
}

// declarar máscara en memoria constante...
// ...

__global__ void Kernel_Convolucion(int * inputArray, int* outputArray, int* mask){

}

__global__ void Kernel_Convolucion_Constante(int * inputArray, int* outputArray){

}

__global__ void Kernel_Convolucion_Shared(int * inputArray, int* outputArray, int* mask){
	
	__shared__ int elements[ CHUNK + 2*(int)(MASK_SIZE/2) ];

	int start	= blockDim.x * blockIdx.x  + threadIdx.x,
		radio	= (int)(MASK_SIZE / 2),
		i		= 0,
		j		= 0,
		maskInd = 0;

	elements[threadIdx.x + radio] = inputArray[start];

	if (threadIdx.x == 0 ) {
		
		if (start == 0 ) {
			for (i = 0 ; i < radio; i++) {
				elements[i] = 0 ;	
			}
			
		} else {
			for(i = 0 ; i < radio; i++){
				elements[ i ] = inputArray[start - radio + i ];
			}
		}
		
	} 
	else if (threadIdx.x == CHUNK - 1) {
		
		if (start == SIZE_X - 1 ){ // Soy el ultimo elemento del array de entrada?
			for (i = 0; i < radio; i++){
				elements[CHUNK + radio + i ] = 0;
			}
		} else { 	
			// Soy el ultimo elemento del bloque
			for (i = 0; i < radio; i++) {
				elements[blockDim.x + radio + i ] = inputArray[start + i + 1];
			}
		}
		
	}
	__syncthreads(); 
	
	
	
	int centro = threadIdx.x + radio,
		min = centro - radio,
		max = centro + radio,
		ac = 0	;
	
	for(i = threadIdx.x ; i <= threadIdx.x + 2*radio; i++){
		
		ac+= elements[ i ] *mask[maskInd]  ;
		//printf(" %d \n" , maskInd);
		maskInd++;
		
	}

	outputArray[start] = ac;
}

void Convolucion_C(int * inputArray, int* ouputArray, int * mask)
{
	int i, j;

	for( i = 0; i<SIZE_X;i++){   
		ouputArray[i] = 0;
		for( j =0; j<MASK_SIZE;j++){      
			int position = i-(int)(MASK_SIZE/2) + j;
			if(position>=0 && position<SIZE_X)
				ouputArray[i] += inputArray[position] * mask[j];
		}       
	}
} 

int main() {

	int* inputArray = (int*)malloc(sizeof(int) * SIZE_X);
	int* outputArray_CPU = (int*)malloc(sizeof(int) * SIZE_X);
	int* outputArray_GPU = (int*)malloc(sizeof(int) * SIZE_X);
	int* mask = (int*)malloc(sizeof(int) * MASK_SIZE); 
	int* mask_dev;
	int i;

	struct timeval a, b,c,d,e;

	// arrays en el device
	int * inputArray_dev;
	int * outputArray_dev;	
	//int* mask_dev;	

	float t_i, t_f, t_sys, diff;

	// memoria para arrays en dispositivo
	cudaMalloc(&inputArray_dev, sizeof(int)*SIZE_X);
	cudaMalloc(&outputArray_dev, sizeof(int) * SIZE_X);
	cudaMalloc(&mask_dev, sizeof(int) * MASK_SIZE);

	cudaCheck();

	for (i = 0; i < SIZE_X; i++){
		inputArray[i] = 1;
		outputArray_CPU[i] = 0;
		outputArray_GPU[i] = 0;
	}		

	//definir una máscara...
	for (i = 0; i < MASK_SIZE; i++) {
		mask[i] = 1;
	}

	// Convolución en CPU...	
	clockStart();
	Convolucion_C(inputArray, outputArray_CPU, mask);
	clockStop("CPU");

	// copiar array de entrada al dispositivo...
	cudaMemcpy(inputArray_dev, inputArray,  sizeof(int) *SIZE_X, cudaMemcpyHostToDevice);
	cudaMemcpy(outputArray_dev, outputArray_GPU,  sizeof(int) *SIZE_X, cudaMemcpyHostToDevice);
	cudaMemcpy(mask_dev, mask, sizeof(int) * MASK_SIZE, cudaMemcpyHostToDevice);

	int cantBloques = SIZE_X / CHUNK;
	int tamBloque = CHUNK;

	
	clockStart();
	Kernel_Convolucion_Shared<<<cantBloques, tamBloque>>>(inputArray_dev, outputArray_dev, mask_dev);
	cudaDeviceSynchronize();
	clockStop("GPU");
	cudaCheck();
 
	// copiar array de salida desde el dispositivo...
	cudaMemcpy(outputArray_GPU,outputArray_dev,sizeof(int) *SIZE_X,cudaMemcpyDeviceToHost);

	// chequear salida...
	for(i = 0; i < SIZE_X; i++){
		if (outputArray_CPU[i] != outputArray_GPU[i]){
			printf("outputArray_CPU[%d] != outputArray_GPU[%d] \n",i,i);
			break;
		}
	}	

	// if (i == SIZE_X -1 )	printf("OK !!" );

	// liberar memoria cpu...
	free(inputArray);
	free(outputArray_CPU);
	free(outputArray_GPU);
	free(mask);

	// liberar memoria dispositivo...
	cudaFree(inputArray_dev);
	cudaFree(outputArray_dev);
	cudaFree(mask_dev);

	return 0;
}

