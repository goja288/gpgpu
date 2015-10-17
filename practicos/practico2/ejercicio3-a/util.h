#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>

#define chunk 256

#define CANT_REPETICIONES 1

float sum_matrix(const float *M, int m, int n=1);
void print_matrix(const float *M, int width);
void clean_matrix(float *M, int width);
void init_matrix(float *M, int width);
void clockStart();	
void clockStop(const char * str);