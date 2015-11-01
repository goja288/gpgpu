#include <stdio.h>



void generarMatriz(float* m, int width, int height){
	for(int i=0; i< width*height; i++){
		m[i] = 1;
	}
}

void imprimirMatriz(float* m, int width, int height){
	int i, j;

	
	for(j =0; j < height; j++){
		for(i=0;i < width; i++){
			fprintf(stderr, "%.2f  ", m[ j*width + i] );
		}
		fprintf(stderr, "\n" );
	}

}

void imprimirMatrizInt(int* m, int width, int height){
	int i, j;
	
	for(j =0; j < height; j++){
		for(i=0;i < width; i++){
			fprintf(stderr, "%d  ", m[ j*width + i] );
		}
		fprintf(stderr, "\n" );
	}

}

bool compareArray(float* a1, float* a2, int width, int height){

	int i =0, tope = width * height;
	for(; i< tope; i++){
		if (a1[i] != a2[i]){
			printf(" FAILS  :(  NO SON LOS MISMO ARREGLOS \n ");
			return false;
		}
	}
	printf(" OK!! :)  SON LOS MISMO ARREGLOS \n ");
	return true;
}

 float* testear(float* matriz, int width, int height, int maskSize){


	int i, 
		j,
		m, 
		n,
		divProm = maskSize * maskSize,
		x, y,
		maskPadding = (int)(maskSize / 2);

	float* prom = (float*)malloc(sizeof(float) * width * height);
	float total;

	for( i=0; i< height; i++){

		for(j =0; j < width; j++){

			m = i - maskPadding;
			total =0;
			x = m + maskSize;
			for( ; m < x; m ++){
				n = j - maskPadding;
				y = n + maskSize;
				for(; n < y; n++){
					if (m < 0 || n < 0 || m >= height || n >= width){
						total+=0;
					}else{
						total+= matriz[ m* width + n ];
					}
				}
			}

			prom[width * i + j]  = total / divProm;
		}
	}

	return prom;
}