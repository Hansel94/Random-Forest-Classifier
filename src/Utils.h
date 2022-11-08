#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>

float** get_matrix(int n, int m);
void free_matrix(float** matrix, int n);
float** combine_matrices(float** first, float** second, int n1, int n2, int cols);

double get_accuracy(int n, float* actual, float* prediction);

int contains_int(int* arr, int n, int val);
int contains_float(float* arr, int n, float val);

struct dimensions
{
    int rows;
    int cols;
};


#endif
