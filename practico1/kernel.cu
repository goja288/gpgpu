#include "cuda.h"
#include "cuda_runtime.h"

#include <stdio.h>
#include <iostream>
#include "util.h"

using namespace std;
#define _CHUNK 32 // TAMAÑO DE MATRICES DEBEN SER MULTIPLO DE _CHUNK !!!!!!

//Kernel

// Suma por columnas de una matriz con un solo bloque
__global__ void MatrixSumKernel_1(int M, float* A_dev, float* SumPar_dev){

	// Pvalue es usado para el valor intermedio
	float Pvalue = 0;

	int offset = threadIdx.y * M;

	for (int k = 0; k < M; k++) {
		Pvalue = Pvalue + A_dev[offset+k];
	}

	SumPar_dev[threadIdx.y] = Pvalue;

}

// Ejercicio 1
__global__ void MatrixSumKernel_2(int M, float* A_dev, float* SumPar_dev) {

	float tmpValue = 0;
	int index = blockIdx.x * blockDim.x + threadIdx.x;


	int offset = index* M;
	int k = 0;

	for (; k< M;k++) {
		tmpValue+= A_dev[offset + k];
	}
	SumPar_dev[index] = tmpValue;
	

}

// Ejericio 2
__global__ void SumaColMatrizKernel2(int M, float* A_dev, float* SumPar_dev){

	float tmpValue = 0;
	int size = M / blockDim.x;
	int start = blockIdx.y * M + threadIdx.x * size;
	int end = start + size;
	int k = start;
	for (; k < end; k++) {
		tmpValue = tmpValue + A_dev[k];
	}

	atomicAdd( & (SumPar_dev[ blockIdx.y ]) , tmpValue );

}

// Ejecicio 3.a
__global__ void SumaColMatrizKernel3A(int M,float* A_dev, float* SumPar_dev){
	
	float tmpValue = 0;
	int columna = blockIdx.y * M;
	int chunk2 = gridDim.y / blockDim.x ;//blockDim.x;
	
	//  Coalesced 
	for (int i = 0; i < chunk2; i++) {
		tmpValue += A_dev[ columna + threadIdx.x + i*chunk2 ];
	}

	atomicAdd( &( SumPar_dev[ blockIdx.y ] ), tmpValue );
	
}

// Ejercicio 3.b
__global__ void SumaColMatrizKernel3B(int M,float* A_dev, float* SumPar_dev){

	float tmpValue = 0;

	int start =  blockDim.x * blockIdx.x  + threadIdx.x * M + blockIdx.y*blockDim.x;
	int end = start + blockDim.x;
	
	for(int k = start; k < end ; k++){
		tmpValue = tmpValue + A_dev[k];
		
	}
	
	
	atomicAdd( &( SumPar_dev[  blockDim.x * blockIdx.x  + threadIdx.x  ] ), tmpValue );

}




//extern "C"
float sumaColMatriz(int M, int N, float * A_hst, int algoritmo) {

	size_t size = M * N * sizeof(float);
	size_t size2 = N*sizeof(float);

	float* A_dev, *SumPar_dev;

	float *SumPar_hst = (float *)malloc(N*sizeof(float));

	// Allocate en device
	cudaMalloc(&A_dev, size);
	cudaMalloc(&SumPar_dev, size2);

	// Inicializo matrices en el device
	clockStart();
	cudaMemcpy(A_dev, A_hst, size, cudaMemcpyHostToDevice);
	cudaMemset(SumPar_dev,0, size2);
	clockStop("transf CPU -> GPU");

	clockStart();
	// 
	switch(algoritmo) {
		
		case 1: {// ejemplo dado

			//Configurar la grilla
			dim3 tamGrid (1, 1); //Grid dimensi? 1x1, Tiene un bloque de ejecucion
			dim3 tamBlock(1, N); //Block dimensi? 1 x N, Hay N threads por bloque, threadId.x = 0, threadId.y esta en [0 .. N -1]

			for(int i = 0; i < CANT_REPETICIONES; i++)
				MatrixSumKernel_1<<<tamGrid, tamBlock>>>(M, A_dev, SumPar_dev);

			break;
		}
		case 2: { // Ejercicio 1
			int _chunk2 = 32; // DESVENTAJA: Threads sin utilizar, hacer chequeos extras
			dim3 tamGrid( (int)( N / _CHUNK)  , 1);

			// dispongo bloques horizontalmente
			dim3 tamBlock( _CHUNK,  1);
			
			// dispongo threads horizontalmente
			for(int i = 0; i < CANT_REPETICIONES; i++)
				MatrixSumKernel_2<<<tamGrid, tamBlock>>>(N, A_dev, SumPar_dev);
			
			break;
		}
		case 3: { // Ejercicio 2
			

			dim3 tamGrid(1, N); //Grid dimensión, N bloques
			dim3 tamBlock(_CHUNK,1,1); //Block dimensión

			// lanzamiento del kernel
			for(int i = 0; i < CANT_REPETICIONES; i++) 
				SumaColMatrizKernel2<<<tamGrid, tamBlock>>>(M, A_dev, SumPar_dev);
			

			
			break;
		}
		case 4: { // Ejercicio 3.a

			dim3 tamGrid(1, N); //Grid dimensión, N bloques
			dim3 tamBlock(_CHUNK,1,1); //Block dimensión
		
			// lanzamiento del kernel
			for (int i = 0; i < CANT_REPETICIONES; i++)
				SumaColMatrizKernel3A<<<tamGrid, tamBlock>>>(M, A_dev, SumPar_dev);
		
			break;
		}
		case 5: { // Ejercicio 3.b

			int newChunk = 32;
			dim3 tamGrid(  (int) (M / newChunk), (int) (M / newChunk) ); //Grid dimensión, N bloques
			dim3 tamBlock(newChunk,1,1); //Block dimensión
		
			// lanzamiento del kernel
			for (int i =0; i<CANT_REPETICIONES ; i++)
				SumaColMatrizKernel3B<<<tamGrid, tamBlock>>>(M, A_dev, SumPar_dev);
		
			break;
		}
	}

	cudaDeviceSynchronize();
	clockStop("kernel");

	// Traer resultado;
	clockStart();
	cudaMemcpy(SumPar_hst, SumPar_dev, size2, cudaMemcpyDeviceToHost);
	clockStop("transf CPU <- GPU");

	// Sumar el vector de resultados parciales;
	float total = 0.0;
	total = sum_matrix(SumPar_hst,N);

	free(SumPar_hst);

	// Free matrices en device
	cudaFree(A_dev); cudaFree(SumPar_dev);

	return total;


}


int main(int argc, char** argv) {

	if (argc < 3) {
		printf("Uso:\nMatSum n algo(1:3)");
		exit(0);
	}

	int n = atoi(argv[1]);
	int algo = atoi(argv[2]);

	float *A = (float *)malloc(n*n*sizeof(float));

	init_matrix(A,n);

	clockStart();
	float result_ref = sum_matrix(A,n,n);
	clockStop("CPU");

	float result_gpu = sumaColMatriz(n,n,A,algo);

	printf("res_cpu - %f / res_gpu - %f \n",result_ref, result_gpu);

	if (result_gpu == result_ref)
		printf("\n\nResultado OK!! :)\n\n\n");
	else
		printf("\n\Segui participando\n\n\n");

	free(A);

	char c = getchar();

	printf("%i",c);

	return 0;
}


