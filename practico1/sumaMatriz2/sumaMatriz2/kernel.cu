
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "util.h"

using namespace std;


//Kernel

// Suma por columnas de una matriz con un solo bloque
__global__ void MatrixSumKernel_1(int M, double* A_dev, double* SumPar_dev){

	// Pvalue es usado para el valor intermedio
	double Pvalue = 0;
  
	int offset = threadIdx.y * M;
  
	for (int k = 0; k < M; k++) {
		Pvalue = Pvalue + A_dev[offset+k];
	}
	
	SumPar_dev[threadIdx.y] = Pvalue;

}

__global__ void MatrixSumKernel_2(int M, double* A_dev, double* SumPar_dev){

}

__global__ void MatrixSumKernel_3(int M,int N,double* A_dev, double* SumPar_dev){

}

__global__ void MatrixSumKernel_4(int M,int N,double* A_dev, double* SumPar_dev){

}


//extern "C" 
double sumaColMatriz(int M, int N, double * A_hst, int algoritmo){


	size_t size = M * N * sizeof(double);
	size_t size2 = N*sizeof(double);

	double* A_dev, *SumPar_dev;

	double *SumPar_hst = (double *)malloc(N*sizeof(double));

	// Allocate en device 
	cudaMalloc(&A_dev, size);
	cudaMalloc(&SumPar_dev, size2);

	// Inicializo matrices en el device
	clockStart();
	cudaMemcpy(A_dev, A_hst, size, cudaMemcpyHostToDevice);
	cudaMemset(SumPar_dev,0, size2);
	clockStop("transf CPU -> GPU");

	clockStart();


	switch(algoritmo){
		case 1:{
				
			//Configurar la grilla
			dim3 tamGrid (1, 1); //Grid dimensión
			dim3 tamBlock(1, N); //Block dimensión

			for(int i = 0; i < CANT_REPETICIONES; i++)
				MatrixSumKernel_1<<<tamGrid, tamBlock>>>(M, A_dev, SumPar_dev);

			break;

		}case 2:{

				
			printf("\n\nNo implementadoooooo!! :)\n\n\n");

			//...
			break;

		}case 3:{
				
			printf("\n\nNo implementadoooooo!! :)\n\n\n");

			//...
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
	double total = 0.0;
	total = sum_matrix(SumPar_hst,N);

	free(SumPar_hst);
	// Free matrices en device
	cudaFree(A_dev); cudaFree(SumPar_dev); 

	return total;


}


int main(int argc, char** argv){

	if (argc < 3){
		printf("Uso:\nMatSum n algo(1:3)");
		exit(0);
	}
	
	int n= atoi(argv[1]);
	int algo = atoi(argv[2]);
	
	double *A = (double *)malloc(n*n*sizeof(double));

	init_matrix(A,n);

	clockStart();
	double result_ref = sum_matrix(A,n,n);
	clockStop("CPU");

	printf("algo - %i \n",algo);

	double result_gpu = sumaColMatriz(n,n,A,algo);

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


