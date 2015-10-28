
#define CHUNK 32
#define MASKSIZE 5

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "CImg.h"
#include "mock.h"

using namespace cimg_library;
using namespace std;

__global__ void kernel(float* input, float* ouput, int imgWidth, int imgHeight) {

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
	/**************************************************************************/
	int i, 
		j,
		width,
		height;

	unsigned int img_matrix_size;

	float* img_matrix,
		 * input_img_dev,
		 * output_img_dev, 
		 * img_matrix_output;

	float tmp;

	/*************************** CARGA DE IMAGEN *****************************/

	CImg<float> image("img\\fing.pgm");

	width =   image.width();
	height = image.height();

	img_matrix_size = width * height * sizeof(float);
	img_matrix = image.data();

	//float* img_matrix = (float*)malloc( img_matrix_size );

	//generarMatriz(img_matrix, width, height);
	//imprimirMatriz(img_matrix, width, height);

	/*************************** PREPARO TEST ***********************************/

	float* testAVG = testear(img_matrix, width, height, MASKSIZE);
	//imprimirMatriz(testAVG, width, height);

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

	kernel <<< gridDimension, blockDimension>>>(input_img_dev, output_img_dev, width, height);
	cudaCheck();
	cudaDeviceSynchronize();

	cudaMemcpy( img_matrix_output, output_img_dev, img_matrix_size, cudaMemcpyDeviceToHost);
	compareArray( testAVG, img_matrix_output, width, height);
	//imprimirMatriz(img_matrix_output, width, height);

	/***************************** CREO IMAGEN *****************************/

	CImg<float> result(width,height,1, 3, 1);
	for(i=0; i< height; i++){
		for(j=0; j< width; j++){
			tmp =  img_matrix_output[width * i + j];
			result(j, i, 0) = tmp;
			result(j, i, 1) = tmp;
			result(j, i, 2) = tmp;
		}
	}

	
	//result.save_png("resultado.png");
	
	CImgDisplay main_disp(result,"FINAL");
	CImgDisplay original(image, "ORIGINAL");

	while (!main_disp.is_closed()) {
		main_disp.wait();
	}
	
	while (!original.is_closed()) {
		original.wait();
	}

	cudaFree(input_img_dev);
	cudaFree(output_img_dev);

	//free(img_matrix);
	free(testAVG);
	free(img_matrix_output);

	//getchar();


	return 0;
}

