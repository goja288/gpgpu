#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "CImg.h"

using namespace cimg_library;
using namespace std;

#define CHUNK 32
#define MASKSIZE 5

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

    return 0;
}

