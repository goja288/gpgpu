#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "CImg.h"
#include "mock.h"

using namespace cimg_library;
using namespace std;

#define CHUNK 32
#define MASKSIZE 5

__global__ void kernelParte2(float* input, float* ouput, int imgWidth, int imgHeight) {

	__shared__ float maskMemShared[ ((int)(MASKSIZE / 2) * 2 + CHUNK) * ((int)(MASKSIZE / 2) * 2 + CHUNK)];

	/******************* CARGA DE MEMORIA COMPARTIDA *************/

	int maskPadding = (int)(MASKSIZE / 2),
	    maskDim  = maskPadding * 2 + CHUNK, // maskDim x maskDim

	    column =  threadIdx.x,
	    row = threadIdx.y,

	    globalColumn = blockIdx.x * blockDim.x  + column,
	    globalRow = blockIdx.y * blockDim.y  + row,

	    startColumn = globalColumn - maskPadding,
	    startRow    = globalRow - maskPadding,

	    iColumn = startColumn,
	    iRow = 0,

	    iMask = startRow,
	    jMask = startColumn,

	    i, iTope,
	    j, jTope;

	float avg = 0;

	for ( iRow  = row; iRow < maskDim; iRow += CHUNK ) {
		jMask = startColumn;
		for ( iColumn = column ; iColumn < maskDim; iColumn += CHUNK) {

			if (jMask < 0 || iMask < 0 || jMask >= imgWidth || iMask >= imgHeight) {
				maskMemShared[ iRow * maskDim + iColumn ] = 0;
			} else {
				maskMemShared[ iRow * maskDim + iColumn] =  input[imgWidth * iMask + jMask];
			}
			jMask += CHUNK;
		}
		iMask += CHUNK;
	}

	__syncthreads();
	/*
	if (column == 0 && row == 0 && blockIdx.x == 2 && blockIdx.y == 0) {

		printf("%d  %d  \n ", globalColumn, globalRow);

		for (int i = 0; i < maskDim ; i++) {
			for (int j = 0; j < maskDim ; j++) {
				printf("%f ", maskMemShared[i * maskDim + j]);
			}
			printf("\n");
		}

	}
	*/


	/************** CUENTAS ***************/

	if (globalColumn < imgWidth  && globalRow < imgHeight) {

		i = row;
		j = column;

		iTope = i + MASKSIZE;
		jTope = j + MASKSIZE;

		for ( ; i < iTope; i++) {
			for ( j = column; j < jTope; j++) {
				avg += maskMemShared[ i * maskDim + j];
			}
		}
		startRow += maskPadding;
		startColumn += maskPadding;

		ouput[imgWidth * startRow + startColumn] = avg / (MASKSIZE * MASKSIZE);

	}
}

__global__ void kernelParte3(float* imagenPromedio, int* a_centros, float voronoiOutput, int width, int height) {

}

// Retorna un array con la posicion de cada centro
int* sorteoCentros(int cantCentros, int width, int height) {

	int* a_centros = (int*)malloc(sizeof(int) * width * height);
	for (int i = 0; i < width * height; i++) {
		a_centros[i] = 0;
	}

	int centro;
	if (cantCentros >= width * height) {
		// te fuiste de tema tenes mas centro que pixeles
		printf("Ehhh mmm te excediste un poco con la cantidad de centros\n ");
	}
	else {
		int c = 0;
		int maxIteraciones = cantCentros * 10 ;
		while (c < maxIteraciones && cantCentros > 0) {
			centro = (int) (rand() % (width * height));
			//printf("aaa %d\n", centro);
			
			// Si no habia asignado un centro lo agrego, si ya habia intento sortear de nuevo
			if (a_centros[centro] == 0) { 
				a_centros[centro] = 1;
				cantCentros--;
			}
			c++;
		}
	}

	/** debug **/

	int cantCentrosAux = 0;
	printf("Centros asignados en: ");
	for(int aux = 0; aux < width*height; aux++) {
		if (a_centros[aux] == 1) {
			printf("%d,",aux);
			cantCentrosAux++;
		}
	}
	printf("\n\nCant Centros asignados : %d\n", cantCentrosAux);
	/** fin debug **/

	return a_centros;

}

/**
 * Funciones auxiliares
 */
void cudaCheck()
{
	cudaError_t cudaError;

	cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		printf("  cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
	}
}

int main()
{

	float* input_img_dev,
		 * output_img_dev, 
		 * img_matrix_output;

	// Cargamos la imagen original
	CImg<float> image("img/fing.pgm");

	int width = image.width();
	int height = image.height();

	unsigned int img_matrix_size = width * height * sizeof(float);
	float* img_matrix = image.data();

	sorteoCentros(42,width,height);

	/**
	 * Parte 1 - Secuencial
	 */
	
	// Creo la matriz a retornar 
	float* matrizVoronoi = (float*) malloc(sizeof(float) * width * height);
	int ventana = (int) MASKSIZE / 2; // para obtener hacia los costados

	// Recorro toda la imagen
	for (int fila = 0; fila < height; fila++) {
		for (int columna = 0; columna < width; columna++) {
			
			int cantPixeles = 0;
			float sumaColoresPixeles = 0;

			// Recorro la ventana para obtener el promedio
			int inicioX = columna - ventana;
			int inicioY = fila - ventana;
			int finX =  columna + ventana;
			int finY = fila + ventana;
			int x,y;

			// Recorro por ventana
			for (x = inicioX; x < finX; x++) {
				for (y = inicioY; y < finY; y++) {
					// Para filtar los bordes de la imagen
					if ( (x > 0) && (x < width) && (y > 0) && (y < height)) {
						 cantPixeles++;
						 sumaColoresPixeles += img_matrix[y * width + x];
					}
				}
			}

			// Cargo a cada centro el promedio
			matrizVoronoi[fila * width + columna] = sumaColoresPixeles / cantPixeles;

		}
	}
	
	// Imagen resultante
	float tmp;
	CImg<float> result(width,height,1, 3, 1);
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			tmp =  matrizVoronoi[width * i + j];
			result(j, i, 0) = tmp;
			result(j, i, 1) = tmp;
			result(j, i, 2) = tmp;
		}
	}

	CImgDisplay main_disp(image,"FingOriginal");
	CImgDisplay secuencial(result,"Secuencial");
		
	// Hasta que no se cierre la imagen original mostrar las 2 imagenes
	while (!main_disp.is_closed()) {
		main_disp.wait();
	}

	/**
	 * FIN Parte 1 - Secuencial
	 */


	/**
	 * Parte 2 
	 */
	
	/*************************** PREPARO TEST ***********************************/

	float* testAVG = testear(img_matrix, width, height, MASKSIZE);

	/*************************** RESERVA DE MEMORIA *****************************/

	input_img_dev;
	output_img_dev;
	img_matrix_output = (float*)malloc( img_matrix_size );

	cudaMalloc(&input_img_dev, img_matrix_size);
	cudaMalloc(&output_img_dev, img_matrix_size);


	cudaMemset(output_img_dev, 0, img_matrix_size);
	cudaMemcpy(input_img_dev, img_matrix, img_matrix_size, cudaMemcpyHostToDevice);

	/************************* AJUSTE DE INVOCACION *********************/

	dim3 gridDimension( (int)( width / CHUNK) + (width % CHUNK == 0 ? 0 : 1), (int)(height / CHUNK ) + (height % CHUNK == 0 ? 0 : 1) );
	dim3 blockDimension(CHUNK, CHUNK);

	kernelParte2 <<< gridDimension, blockDimension>>>(input_img_dev, output_img_dev, width, height);
	cudaCheck();
	cudaDeviceSynchronize();

	cudaMemcpy(img_matrix_output, output_img_dev, img_matrix_size, cudaMemcpyDeviceToHost);
	compareArray(testAVG, img_matrix_output, width, height);

	//result.save_png("resultado.png");
	
	CImgDisplay parte2(result,"Parte2");
	
	while (!parte2.is_closed()) {
		parte2.wait();
	}

	cudaFree(input_img_dev);
	cudaFree(output_img_dev);

	//free(img_matrix);
	free(testAVG);
	free(img_matrix_output);
	
	/**
	 * Fin Parte 2 
	 */

	/**
	 * Parte 3
	 */

	/**
	 * FIN Parte 3
	 */


    return 0;
}

