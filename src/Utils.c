#include "Utils.h"


//Returns a 2d float matrix
float** get_matrix(int n, int m)
{
    float** matrix = (float**)malloc(n*sizeof(float*));
    for(int i = 0; i < n; i++){
    	matrix[i] = (float*)malloc(m*sizeof(float));
    }
    
    return matrix;
}

//Frees matrix
void free_matrix(float** matrix, int rows)
{
    for(int i = 0; i < rows; i++){
    	free(matrix[i]);
    }
    free(matrix);
}

//Combines matrix
float** combine_matrices(float** first, float** second, int n1, int n2, int cols)
{
    float** combined = (float**) malloc((n1 + n2) * sizeof(float) * cols);
    int row_index = 0;
    for(int i=0; i < n1; i++)
    {
        float* row = first[i];
        combined[row_index] = row;
        row_index++;
    }
    for(int j=0; j < n2; j++)
    {
        float* row = second[j];
        combined[row_index] = row;
        row_index++;
    }
    return combined;
}

//Calculates the predictions accuracy
double get_accuracy(int n, float* actual, float* prediction)
{
    int correct = 0;
    for(int i=0; i < n; i++)
    {
        if(actual[i] == prediction[i]) correct++;
    }
    return (correct * 1.0 / n * 1.0) * 1.0;
}

//Array management
//Array of ints contains val
int contains_int(int* arr, int n, int val)
{
    for(int i=0; i < n; i++)
    {
        if(arr[i] == val) return 1;
    }
    return 0;
}
//Array of floats contains val
int contains_float(float* arr, int n, float val)
{
    for(int i=0; i < n; i++)
    {
        if(arr[i] == val) return 1;
    }
    return 0;
}
