#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "CImg.h"

using namespace cimg_library;

#ifndef SIZE_WIN_X
#define SIZE_WIN_X 5
#endif

#ifndef SIZE_WIN_Y
#define SIZE_WIN_Y 5
#endif

void voronoi_CPU (CImg<float> originalImage);

void voronoi_GPU();
