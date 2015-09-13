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

#define CHUNK 16
#define SIZE_X 2048
#define MASK_SIZE 3


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

__global__ void Kernel_Convolucion(int * inputArray, int* outputArray, int* mask, int arraySize, int maskSize){

	int i = 0,
		j = 0,
		position,
		radio = (int)maskSize / 2;

	for (i = - radio; i < (threadIdx.x + radio); i++) {
		if (i >= 0 && i < arraySize ) {                    
			if (blockIdx.x == 1 && threadIdx.x == 1) {
				printf("El primer hilo suma en el indice [%d * %d + %d] * mask[%d + %d]",blockDim.x,blockIdx.x,threadIdx.x,i,i,maskSize);
			}
			outputArray[blockDim.x * blockIdx.x  + threadIdx.x] += inputArray[blockDim.x * blockIdx.x  + threadIdx.x + i] * mask[i + maskSize];
		}
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
	int* mask_dev;	

	float t_i, t_f, t_sys, diff;

	// memoria para arrays en dispositivo
	inputArray_dev = (int*)malloc(sizeof(int) * SIZE_X);
	outputArray_dev = (int*)malloc(sizeof(int) * SIZE_X);
	mask_dev = (int*)malloc(sizeof(int) * MASK_SIZE);


	cudaCheck();

	for (i = 0; i < SIZE_X; i++){
		inputArray[i] = i % 10;
		inputArray_dev[i] = i % 10;
		outputArray_CPU[i] = 0;
		outputArray_GPU[i] = 0;
		//	outputArray_dev[i] = 0; // ????
	}		

	//definir una máscara...
	for (i = 0; i < MASK_SIZE; i++) {
		mask[i] = 1;
		//	mask_dev[i] = 1; // ?????
	}

	// Convolución en CPU...	
	clockStart();
	Convolucion_C(inputArray, outputArray_CPU, mask);
	clockStop("CPU");

	// copiar array de entrada al dispositivo...
	cudaMemcpy(inputArray_dev, inputArray, SIZE_X, cudaMemcpyHostToDevice);
	cudaMemcpy(outputArray_dev, outputArray_GPU, SIZE_X, cudaMemcpyHostToDevice);
	cudaMemcpy(mask_dev, mask, MASK_SIZE, cudaMemcpyHostToDevice);

	// setear en 0 el array de salida en el dispositivo...
	// ...
	//cudaMemset(outputArray_dev, 0, SIZE_X);
		

	// copiar la máscara o setear la máscara en memoria constante (cudaMemcpyToSymbol)
	// ...

	int cantBloques = SIZE_X / CHUNK;
	int tamBloque = CHUNK;

	clockStart();
	Kernel_Convolucion<<<cantBloques, tamBloque>>>(inputArray_dev, outputArray_dev, mask_dev, SIZE_X, MASK_SIZE);
	cudaDeviceSynchronize();
	clockStop("GPU");
 
	// copiar array de salida desde el dispositivo...
	cudaMemcpy(outputArray_GPU,outputArray_dev,SIZE_X,cudaMemcpyDeviceToHost);

	// chequear salida...
	for(i = 0; i < SIZE_X; i++){
		printf("Joaco se la : %d ::: %d \n", outputArray_CPU[i],  outputArray_GPU[i]);
		
		if (outputArray_CPU[i] != outputArray_GPU[i]){
			printf("outputArray_CPU[%d] != outputArray_GPU[%d] \n",i,i);
			break;
		}
	}	

	// liberar memoria cpu...
	free(inputArray);
	free(outputArray_CPU);
	free(outputArray_GPU);
	free(mask);

	// liberar memoria dispositivo...
	cudaFree(inputArray_dev);
	cudaFree(outputArray_dev);
	//cudaFree(outputArray_GPU);
	cudaFree(mask_dev);

	return 0;
}
