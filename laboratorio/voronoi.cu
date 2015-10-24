#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CImg.h"
#include "voronoi.h"
#include <stdio.h>

using namespace cimg_library;

int main()
{
    
	CImg<float> image("img/fing.pgm");
	
	float * img_matrix = image.data();

	size_t size = image.width()*image.height()*sizeof(float);
	
	//filtrar
	//	...
	//

	voronoi_CPU(image);

	/*CImgDisplay main_disp(image,"Fing");
		
	while (!main_disp.is_closed()) {
		main_disp.wait();
	}
	printf("%s\n", "Hi manso....");*/

	return 0;
}


void voronoi_CPU(CImg<float> originalImage) 
{

	//CImg<float> image(originalImage);

	CImgDisplay main_disp(originalImage,"Fing2");
		
	while (!main_disp.is_closed()) {
		main_disp.wait();
	}
	printf("%s\n", "Hi manson....");

	//CImgDisplay main_disp(originalImage,"Fing2");
	// matrix

	// Elegimos k centros de forma aleatoria

	// Recorro toda la matriz original mxn ( o sea la imagen)

	// Creo otra matriz mxn donde a cada elemento se le asigna el valor que dentifique al centro que se encuentra a menor distancia


	// Asigno a cada poligono el color promedio de los pixeles de alrededor del centro del mismo

}

void voronoi_GPU() 
{
	
}