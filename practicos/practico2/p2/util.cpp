
#include "util.h"


__int64 ctr1 = 0, ctr2 = 0, freq = 0;

void clockStart(){
	QueryPerformanceCounter((LARGE_INTEGER *)&ctr1);
}

void clockStop(const char * str){
	
	QueryPerformanceCounter((LARGE_INTEGER *)&ctr2);
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
	printf("%s : %fs\n",str,(ctr2 - ctr1) * 1.0 / freq);
	
}

void print_matrix(const double * M, int width){

	for (int i = 0; i<width; i++){
		for (int j = 0; j<width; j++){
			printf("%.2f ", M[i*width+j]);
		}
		printf("\n");
	}
}

float sum_matrix(const float *M, int m, int n){
	
	//suma de Kahan

	float sum = 0.0;
    float c = 0.0;
	for (int i = 0 ; i < m; i++ ){
		printf("%f , ", M[ i ] );
	}

    for (int i = 0 ; i < m*n ; i++ ){
        float y = M[i] - c ;    
        float t = sum + y;       // sum es grande e y es chico. Al aumentar el exp. de y pierdo los bits menos sig de la mantisa.
        c = (t - sum) - y;        // (t - sum) es lo que se conserva de y; restarle y recupera los bits perdidos. (c queda con signo negativo)
        sum = t;                  // En la proxima iteración se le suman a y los bits recuperados en c. 
	}

	return sum;
}


void clean_matrix(double *M, int width){

	for (int i = 0; i<width*width; i++) M[i]=0;

}

void init_matrix(float *M, int width){
	
	for (int i = 0; i<width*width; i++)	M[i]= i;//rand()%100;
	
}