/*******************/
/** Ejercicio 2 a **/
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

#define CHUNK 1024
#define SIZE_X 1048576
#define MASK_SIZE 7



// Declaramos la memoria constante
__constant__ int mask_dev[MASK_SIZE];

void cudaCheck()
{
  cudaError_t cudaError;

  cudaError = cudaGetLastError();
  if(cudaError != cudaSuccess)
  {
    printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
  }
}

// declarar m�scara en memoria constante...
// ...

__global__ void Kernel_Convolucion(int * inputArray, int* outputArray, int* mask, int arraySize, int maskSize){

	
	 __shared__ int elements[ CHUNK + 2*(int)(MASK_SIZE/2) ];
	int start = blockDim.x * blockIdx.x  + threadIdx.x,
		radio = (int)maskSize / 2,
		i = 0,
		j =0,
		maskInd = 0	;
	
	// me cargo lo que corresponde
	elements[ threadIdx.x + radio] = inputArray[start];
	if (threadIdx.x == 0 ){
		for(i = 0 ; i < radio; i++){
			if ( start == 0 ){
				printf("1 > \n");
				elements[i] = 0 ;
			}else{
				printf("2 > \n");
				elements[ i ] = inputArray[start - radio + i ];
			}
		}
		printf("start %d  \n", start);
	}
	
	if (threadIdx.x == blockDim.x - 1 ){
		for(i = CHUNK ; i < CHUNK + radio; i++){
			if (start == SIZE_X-1 ){
				elements[i] = 0;
			}else{
				elements[i] = inputArray[CHUNK + j];
			}
			j++;
		}
		
	}

	__syncthreads();
	
	
	printf("start %d , element[start] %d\n", start, elements[start]);

	for (i =start - radio; i <= (start + radio); i++) {
		                   
		outputArray[start] += elements[i] * mask[maskInd] ;
		maskInd++;
	}
}

__global__ void Kernel_Convolucion_Constante(int * inputArray, int* outputArray){

}

__global__ void Kernel_Convolucion_Shared(int * inputArray, int* outputArray, int* mask){

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
//	cudaMalloc(&mask_dev, sizeof(int) * MASK_SIZE);

	cudaCheck();

	for (i = 0; i < SIZE_X; i++){
		inputArray[i] = i % 10;
		outputArray_CPU[i] = 0;
		outputArray_GPU[i] = 0;
		//	outputArray_dev[i] = 0; // ????
	}		

	//definir una m�scara...
	for (i = 0; i < MASK_SIZE; i++) {
		mask[i] = 1;
		//	mask_dev[i] = 1; // ?????
	}

	// Convoluci�n en CPU...	
	clockStart();
	Convolucion_C(inputArray, outputArray_CPU, mask);
	clockStop("CPU");

	// copiar array de entrada al dispositivo...
	cudaMemcpy(inputArray_dev, inputArray,  sizeof(int) *SIZE_X, cudaMemcpyHostToDevice);
	cudaMemcpy(outputArray_dev, outputArray_GPU,  sizeof(int) *SIZE_X, cudaMemcpyHostToDevice);
	//cudaMemcpy(mask_dev, mask, sizeof(int) * MASK_SIZE, cudaMemcpyHostToDevice);

	// setear en 0 el array de salida en el dispositivo...
	// ...
	//cudaMemset(outputArray_dev, 0, SIZE_X);
		

	// copiar la m�scara o setear la m�scara en memoria constante (cudaMemcpyToSymbol)
	// ...
	cudaMemcpyToSymbol(mask_dev, mask, sizeof(int) * MASK_SIZE);


	int cantBloques = SIZE_X / CHUNK;
	int tamBloque = CHUNK;

	clockStart();
	Kernel_Convolucion<<<cantBloques, tamBloque>>>(inputArray_dev, outputArray_dev, mask_dev, SIZE_X, MASK_SIZE);
	cudaDeviceSynchronize();
	clockStop("GPU");
 
	// copiar array de salida desde el dispositivo...
	cudaMemcpy(outputArray_GPU,outputArray_dev,sizeof(int) *SIZE_X,cudaMemcpyDeviceToHost);

	// chequear salida...
	for(i = 0; i < SIZE_X; i++){
			printf("%d :: %d \n",outputArray_CPU[i],outputArray_GPU[i]);	
		if (outputArray_CPU[i] != outputArray_GPU[i]){
			printf("outputArray_CPU[%d] != outputArray_GPU[%d] \n",i,i);
			break;
		}
	}	

	if (i == SIZE_X -1 )	printf("OK !!" );

	// liberar memoria cpu...
	free(inputArray);
	free(outputArray_CPU);
	free(outputArray_GPU);
	free(mask);

	// liberar memoria dispositivo...
	cudaFree(inputArray_dev);
	cudaFree(outputArray_dev);
	//cudaFree(outputArray_GPU);
	//cudaFree(mask_dev);

	char enter = getchar();

	return 0;
}