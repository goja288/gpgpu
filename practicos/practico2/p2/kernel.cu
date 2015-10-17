#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdlib.h>
#include <Windows.h>
#include "util.h"

#define CHUNK 256
#define SIZE_X 1048576
#define MASK_SIZE 5


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

__constant__ int mask_dev_const[MASK_SIZE];


__global__ void Kernel_Convolucion(int * inputArray, int* outputArray, int* mask, int arraySize, int maskSize){
	
	int start = blockDim.x * blockIdx.x  + threadIdx.x,
		i = 0,
		maskInd = 0,
		radio = (int)maskSize / 2;

	for (i =start - radio; i <= (start + radio); i++) {
		if (i >= 0 && i < arraySize ) {                    
			outputArray[start] += inputArray[i] * mask[maskInd] ;
		}
		maskInd++;
	}
}

__global__ void Kernel_Convolucion_Constante(int * inputArray, int* outputArray){
	int start = blockDim.x * blockIdx.x  + threadIdx.x,
		i = 0,
		maskInd = 0,
		radio = (int)MASK_SIZE / 2;

	for (i =start - radio; i <= (start + radio); i++) {
		if (i >= 0 && i < SIZE_X ) {              
			outputArray[start] += inputArray[i] * mask_dev_const[maskInd] ;
		}
		maskInd++;
	}
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

	int i;

	struct timeval a, b,c,d,e;

	// arrays en el device
	int * inputArray_dev;
	int * outputArray_dev;	

	int* mask_dev;	

	float t_i, t_f, t_sys, diff;

	// memoria para arrays en dispositivo
	// ...
	cudaMalloc(&inputArray_dev, sizeof(int)*SIZE_X);
	cudaMalloc(&outputArray_dev, sizeof(int) * SIZE_X);
	cudaMalloc(&mask_dev, sizeof(int) * MASK_SIZE);

	cudaCheck();

	for(i =0; i<SIZE_X;i++){
		inputArray[i] = 1;
		outputArray_CPU[i] = 0;
		outputArray_GPU[i] = 0;
	}		

	//definir una máscara...
	for(i =0; i<MASK_SIZE; i++){
		mask[i] = 1;
	}

	// Convolución en CPU...	
	clockStart();
	Convolucion_C(inputArray, outputArray_CPU, mask);
	clockStop("CPU");

	// copiar array de entrada al dispositivo...
	// ...
	clockStart();
	cudaMemcpy(inputArray_dev, inputArray,  sizeof(int) *SIZE_X, cudaMemcpyHostToDevice);
	cudaMemcpy(outputArray_dev, outputArray_GPU,  sizeof(int) *SIZE_X, cudaMemcpyHostToDevice);
	cudaMemcpy(mask_dev, mask, sizeof(int) * MASK_SIZE, cudaMemcpyHostToDevice);
	clockStop("CPU -- > GPU");

	// setear en 0 el array de salida en el dispositivo...
	// ...
	cudaMemset(outputArray_dev, 0, sizeof(int) *SIZE_X);

	// copiar la máscara o setear la máscara en memoria constante (cudaMemcpyToSymbol)
	// ...
	clockStart();
	cudaMemcpyToSymbol(mask_dev_const, mask, sizeof(int) * MASK_SIZE);
	clockStop("CPU -- > GPU, Transferencia de memoria constante.");

	// int cantBloques = ...
	// int tamBloque = ...

	int cantBloques = SIZE_X / CHUNK;
	int tamBloque = CHUNK;

	// clockStart();
	// Kernel_Convolucion<<<cantBloques, tamBloque>>>(inputArray_dev, outputArray_dev, mask_dev);
	// cudaDeviceSynchronize();
	// clockStop("GPU");
	clockStart();

	/************************* INVOCACION DE EJERCICIO 1 **************************************/

	Kernel_Convolucion<<<cantBloques, tamBloque>>>(inputArray_dev, outputArray_dev, mask_dev, SIZE_X, MASK_SIZE);

	/************************* INVOCACION DE EJERCICIO 2 **************************************/

	//Kernel_Convolucion_Constante<<<cantBloques, tamBloque>>>(inputArray_dev, outputArray_dev);


	/************************* INVOCACION DE EJERCICIO 3 **************************************/

	//Kernel_Convolucion_Shared<<<cantBloques, tamBloque>>>(inputArray_dev, outputArray_dev, mask_dev);

	/******************************************************************************************/

	cudaDeviceSynchronize();
	clockStop("GPU");
 
	// copiar array de salida desde el dispositivo...
	// ...
	clockStart();
	cudaMemcpy(outputArray_GPU,outputArray_dev,sizeof(int) *SIZE_X,cudaMemcpyDeviceToHost);
	clockStop("GPU --> CPU");

	// chequear salida...
	for(i = 0; i < SIZE_X; i++){
		if (outputArray_CPU[i] != outputArray_GPU[i]){
			printf("outputArray_CPU[%d] != outputArray_GPU[%d] \n",i,i);
			break;
		}
	}	

	printf("Press Enter...");
	char getEnter = getchar();

	// liberar memoria cpu...
	free(inputArray);
	free(outputArray_CPU);
	free(outputArray_GPU);
	free(mask);

	// liberar memoria dispositivo...
	// ...

	cudaFree(inputArray_dev);
	cudaFree(outputArray_dev);
	cudaFree(mask_dev);

	return 0;
}
