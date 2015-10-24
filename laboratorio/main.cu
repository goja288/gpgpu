
SE BORRA 
/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "CImg.h"
#include "voronoi.h"

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

	CImgDisplay main_disp(image,"Fing");
		
	while (!main_disp.is_closed()) {
		main_disp.wait();
	}
	printf("%s\n", "Hi manso....");

	return 0;
}
*/