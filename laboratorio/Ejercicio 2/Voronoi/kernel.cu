#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "CImg.h"

using namespace cimg_library;
using namespace std;

#define CHUNK 32
#define MASKSIZE 5

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

void imprimirMatriz(float* m, int width, int height){
	int i, j;
	for(j =0; j < height; j++){
		for(i=0;i < width; i++){
			fprintf(stderr, "%.2f  ", m[ j*width + i] );
		}
		fprintf(stderr, "\n" );
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

	//imprimirMatriz(img_matrix,width, height);

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
	 * Parte 3 - Secuencial
	 */

	/**
	 * FIN Parte 3 - Secuencial
	 */


    return 0;
}

