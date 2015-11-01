#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <climits>
#include "CImg.h"
#include "mock.h"

using namespace cimg_library;
using namespace std;

#define CHUNK 32
#define MASKSIZE 5
#define CANT_CENTROS 9999

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

__global__ void kernelParte3(float* input, float* output, int* a_centros, int cantCentros, int width, int height) {

	float distanciaActual = -1;
	float distanciaMinima = INT_MAX; // fixme

	// Cargo en memoria compartida

	unsigned int column =  threadIdx.x,
		row = threadIdx.y,
		globalColumn = blockIdx.x * blockDim.x  + column,
		globalRow = blockIdx.y * blockDim.y  + row;

	unsigned int bestX = -1, 
		bestY = -1;

	

	// Controlo que no me pase
	if (globalColumn < width && globalRow < height) {

		// Me fijo en el pixel que estoy la distancia a cada centro
		for (unsigned int i = 0; i < cantCentros; i++) {
			unsigned int x = a_centros[i];
			unsigned int y = a_centros[i + cantCentros];

			//distanciaActual = sqrt( powf((globalColumn-x),2) + powf((globalRow - y),2));
			distanciaActual = sqrtf( (globalColumn-x) * (globalColumn-x) + (globalRow - y) * (globalRow - y));
			
		
			if ( distanciaActual < distanciaMinima ) {
				bestX = x;
				bestY = y;
				distanciaMinima = distanciaActual;
			}
		}


	
		// Asigno el color del centro al pixel
		output[ width  * globalRow +  globalColumn] = input[ width * bestY +  bestX];

	}



	// Mejora limitar a los centros que busca
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

int main()
{

	float* input_img_dev,
		 * output_img_dev, 
		 * img_matrix_output;

	// Cargamos la imagen original
	CImg<float> image("img/fing.pgm");
	//CImg<float> image("img/fing_xl.pgm");

	int width = image.width();
	int height = image.height();

	unsigned int img_matrix_size = width * height * sizeof(float);
	float* img_matrix = image.data();

	

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
					if ( (x >= 0) && (x < width) && (y >= 0) && (y < height)) {
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

	// Liberamos la memoria del device
	cudaFree(input_img_dev);
	

	//free(img_matrix);
	free(testAVG);
	


	/**
	 * Fin Parte 2 
	 */

	/**
	 * Parte 3
	 */

	float* input_img_parte3_dev,
		 * output_img_parte3_dev;

	int* a_centros_parte3_dev;

	unsigned int centros_size = sizeof(int) * CANT_CENTROS * 2;
	// Sorteamos los centros
	// TODO Cargar en memoria compartida los centros
	
	int* a_centros = sorteoCentros(CANT_CENTROS,width,height);


	// Reservamos memoria
	float* img_matrix_parte3_output = (float*)malloc( img_matrix_size );

	cudaMalloc(&input_img_parte3_dev, img_matrix_size);
	cudaMalloc(&output_img_parte3_dev, img_matrix_size);
	cudaMalloc(&a_centros_parte3_dev, centros_size);

	cudaMemset(output_img_parte3_dev, 0, img_matrix_size);
	cudaMemcpy(a_centros_parte3_dev, a_centros, centros_size, cudaMemcpyHostToDevice);

	// Llamamos al otro kernel
	dim3 gridDimensionParte3( (int)( width / CHUNK) + (width % CHUNK == 0 ? 0 : 1), (int)(height / CHUNK ) + (height % CHUNK == 0 ? 0 : 1) );
	dim3 blockDimensionParte3(CHUNK, CHUNK);

	kernelParte3<<<gridDimensionParte3, blockDimensionParte3>>>(output_img_dev, output_img_parte3_dev,a_centros_parte3_dev,CANT_CENTROS,width,height);
	cudaCheck();
	cudaDeviceSynchronize();

	cudaMemcpy(img_matrix_parte3_output, output_img_parte3_dev, img_matrix_size, cudaMemcpyDeviceToHost);
	
	

	// Muestro la imagen de la parte 3

	CImg<float> parte3(width,height,1, 3, 1);
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			tmp =  img_matrix_parte3_output[width * i + j];
			parte3(j, i, 0) = tmp;
			parte3(j, i, 1) = tmp;
			parte3(j, i, 2) = tmp;
		}
	}
		
	//result.save_png("resultado.png");
	
	CImgDisplay parte3_disp(parte3,"Parte3");
	while (!parte3_disp.is_closed()) {
		parte3_disp.wait();
	}


	// Liberamos la memoria 
	free(img_matrix_parte3_output);
	

	/**
	 * FIN Parte 3
	 */

	// Liberamos memoria del device
	cudaFree(input_img_parte3_dev);
	cudaFree(output_img_parte3_dev);
	cudaFree(a_centros_parte3_dev);
	cudaFree(output_img_dev);

	// Liberamos lo que quedo
	free(img_matrix_output);

    return 0;
}

