#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>

#define chunk 256

#define CANT_REPETICIONES 1000

double sum_matrix(const double *M, int m, int n=1);
void print_matrix(const double *M, int width);
void clean_matrix(double *M, int width);
void init_matrix(double *M, int width);
void clockStart();	
void clockStop(const char * str);