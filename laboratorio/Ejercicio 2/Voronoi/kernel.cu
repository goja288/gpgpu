#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <climits>
#include "CImg.h"
#include "mock.h"

using namespace cimg_library;

#define CHUNK 32
#define MASKSIZE 5
#define CANT_CENTROS 5000

void kernelParte1_Secuencial(float* img_matrix, float* output, int imgWidth, int imgHeight){

	int ventana = (int) MASKSIZE / 2; // para obtener hacia los costados

	// Recorro toda la imagen
	for (int fila = 0; fila < imgHeight; fila++) {
		for (int columna = 0; columna < imgWidth; columna++) {
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
					if ( (x >= 0) && (x < imgWidth) && (y >= 0) && (y < imgHeight)) {
						 cantPixeles++;
						 sumaColoresPixeles += img_matrix[y * imgWidth + x];
					}
				}
			}
			// Cargo a cada centro el promedio
			output[fila * imgWidth + columna] = sumaColoresPixeles / cantPixeles;
		}
	}

}

__global__ void kernelParte2(float* input, float* ouput, int imgWidth, int imgHeight) {

	__shared__ float maskMemShared[ ((int)(MASKSIZE / 2) * 2 + CHUNK) * ((int)(MASKSIZE / 2) * 2 + CHUNK)];

	// CARGA DE MEMORIA COMPARTIDA

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
	// CUENTAS

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


__global__ void kernelParte3(float* input, float* output, int* a_centros, int width, int height) {

	float distanciaActual = -1,
		distanciaMinima = INT_MAX,
		tmpOp, 
		op1, 
		op2; // fixme

	// Cargo en memoria compartida
	unsigned int
		
		column =  threadIdx.x,
		row = threadIdx.y,

		globalColumn = blockIdx.x * blockDim.x  + column,
		globalRow = blockIdx.y * blockDim.y  + row,
		
		bestX = 0, 
		bestY = 0,
		x,
		y,
		i=0;

	// Controlo que no me pase
	if (globalColumn < width && globalRow < height) {

		// Me fijo en el pixel que estoy la distancia a cada centro
		for (; i < CANT_CENTROS; i++) {
			x = a_centros[i];
			y = a_centros[i + CANT_CENTROS];
			tmpOp = (globalRow-y)*(globalRow-y) + (globalColumn - x)*(globalColumn - x);
			distanciaActual = std::sqrt( tmpOp );
			
			if ( distanciaActual < distanciaMinima ) {
				bestX = x;
				bestY = y;
				distanciaMinima = distanciaActual;
			}
		}
		// Asigno el color del centro al pixel
		output[ width  * globalRow +  globalColumn] = input[ width * bestY +  bestX];

	}
}

// Retorna un array con la posicion de cada centro
int* sorteoCentros(int cantCentros, int width, int height) {
	
	if (cantCentros >= width * height) {	
		// te fuiste de tema tenes mas centro que pixeles
		printf("Ehhh mmm te excediste un poco con la cantidad de centros\n ");
		return 0;
	}
	else {
		int* a_centros = (int*)malloc(sizeof(int) * cantCentros * 2);
		memset(a_centros,0,sizeof(int) * cantCentros * 2);
		int centroX, centroY;

		for (int i = 0; i < cantCentros; i++) {
			centroX = (int) (rand() % (width));
			centroY = (int) (rand() % (height));
			
			a_centros[i] = centroX;
			a_centros[i + cantCentros] = centroY;

		}


		return a_centros;
	}

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

void showImage(float* img, bool wait, int width, int height, char* title){
	
	int i, j;
	CImg<float> imgDisplay(width,height,1, 3, 1);
	float tmp;

	for(i=0; i< height; i++){
		for(j=0; j< width;j++){
			tmp =  img[width * i + j];
			imgDisplay(j, i, 0) = tmp;
			imgDisplay(j, i, 1) = tmp;
			imgDisplay(j, i, 2) = tmp;
		}
	}

	
	if (wait){

		CImgDisplay disp(imgDisplay,title);

		while (!disp.is_closed()) {
			disp.wait();
		}
	}
}

int main()
{
	// Cargamos la imagen original
	CImg<float> image("img/fing.pgm");
	//CImg<float> image("img/fing_xl.pgm");

	int width = image.width();
	int height = image.height();

	unsigned int img_matrix_size = width * height * sizeof(float);
	float* img_matrix = image.data();
	
	float tmp;

	// ****************** PARTE 1 ********************** //

	float *matrizVoronoi = (float*)malloc(img_matrix_size);
	float *output_parte1 = (float*) malloc(img_matrix_size);

	kernelParte1_Secuencial(img_matrix, output_parte1, width, height);

	// ****************** PARTE 2 ********************** //

	// RESERVA DE MEMORIA

	float* input_img_parte2_dev;
	float* output_img_parte2_dev;
	float* output_parte2 = (float*)malloc(img_matrix_size);
	
	
	cudaMalloc(&input_img_parte2_dev, img_matrix_size);
	cudaMalloc(&output_img_parte2_dev, img_matrix_size);

	cudaMemset(output_img_parte2_dev, 0, img_matrix_size);
	cudaMemcpy(input_img_parte2_dev, img_matrix, img_matrix_size, cudaMemcpyHostToDevice);

	// AJUSTE DE INVOCACION

	dim3 gridDimension( (int)(width / CHUNK) + (width % CHUNK == 0 ? 0 : 1), (int)(height / CHUNK ) + (height % CHUNK == 0 ? 0 : 1) );
	dim3 blockDimension(CHUNK, CHUNK);

	kernelParte2 <<< gridDimension, blockDimension>>>(input_img_parte2_dev, output_img_parte2_dev, width, height);
	cudaCheck();
	cudaDeviceSynchronize();
	cudaMemcpy(output_parte2, output_img_parte2_dev, img_matrix_size, cudaMemcpyDeviceToHost);

	// *********************** PARTE 3 ************************* //
	unsigned int centros_size = sizeof(int) * CANT_CENTROS * 2;

	// Sorteamos los centros
	
	int* a_centros = sorteoCentros(CANT_CENTROS,width,height);

	// Reservamos memoria
	float* img_matrix_parte3_output = (float*)malloc( img_matrix_size );

	float* output_img_parte3_dev;
	int* a_centros_parte3_dev;
	
	cudaMalloc(&output_img_parte3_dev, img_matrix_size);
	cudaMalloc(&a_centros_parte3_dev, centros_size);

	cudaMemset(output_img_parte3_dev, 0, img_matrix_size);
	cudaMemcpy(a_centros_parte3_dev, a_centros, centros_size, cudaMemcpyHostToDevice);
	

	// Llamamos al otro kernel
	dim3 gridDimensionParte3( (int)( width / CHUNK) + (width % CHUNK == 0 ? 0 : 1), (int)(height / CHUNK ) + (height % CHUNK == 0 ? 0 : 1) );
	dim3 blockDimensionParte3(CHUNK, CHUNK);

	kernelParte3<<<gridDimensionParte3, blockDimensionParte3>>>(output_img_parte2_dev, output_img_parte3_dev,a_centros_parte3_dev,width,height);
	cudaCheck();
	cudaDeviceSynchronize();

	cudaMemcpy(img_matrix_parte3_output, output_img_parte3_dev, img_matrix_size, cudaMemcpyDeviceToHost);
	
	

	// ******************* MUESTRO IMAGENES ***********************

	// ORIGINAL
	//showImage(img_matrix, true, width, height, "ORIGINAL");

	// PARTE-1 , SECUENCIAL

	//showImage(output_parte1, true, width, height, "PROMEDIO SECUENCIAL");

	// PARTE-2, PROMEDIO

	showImage(output_parte2, true, width, height, "PROMEDIO CUDA");

	// PARTE-3, CENTROS

	showImage(img_matrix_parte3_output, true, width, height, "CENTROS CUDA");

	// LIBERAR MEMORIA

	// PARTE 1

	free(matrizVoronoi);
	free(output_parte1);

	// PARTE 2

	cudaFree(input_img_parte2_dev);
	cudaFree(output_img_parte2_dev);
	free(output_parte2);

	// PARTE 3
	
	cudaFree(output_img_parte3_dev);
	cudaFree(a_centros_parte3_dev);


	free(img_matrix_parte3_output);
	free(a_centros);


    return 0;
}

