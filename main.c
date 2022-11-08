#include <stdio.h>
#include <string.h>
#include <time.h>
#include "src/Utils.h"
#include "src/Predictions.h"

#define BUFFER_SIZE 1024

struct dimensions get_csv_dimensions(FILE *file); //Function required for getting data dimensions from a csv file
float** read_data(FILE *csv, struct dimensions csv_dim); //Function required for reading and storing data into a matrix

// Getting data dimensions from a csv file
struct dimensions get_csv_dimensions(FILE *file){
    const char *delimiter = ","; //Delimiter character
    char buffer[BUFFER_SIZE]; //Variable that will store rows from csv file
    char *token; //Variable that will store a value (column) from a row, delimited by a comma

    int rows_count = 0; //Counter for rows
    int cols_count = 0; //Counter for columns

    // fgets stores into buffer variable a string of maximum BUFFER_SIZE characters from file, until a "\n" is found
    while (fgets(buffer, BUFFER_SIZE, file) != NULL)
    {
        token = strtok(buffer, delimiter); //Tokenizes the buffer string, by the delimiter chosen

        while (token != NULL)
        {
            if (strstr(token, "\n") != NULL) //Searches a \n into the token, returns NULL if it's not found
            {
                rows_count++; //If \n is found, then we count a new row
            }
            cols_count++; //We count every single token found
            token = strtok(NULL, delimiter); //For keep advancing through the buffer, it's passed NULL as argument instead of the string
        }
    }
    
    fseek(file, 0, SEEK_SET); //It places the pointer to the file start
    cols_count /= rows_count; //For getting number of columns, total number of tokens found is divided by total number of rows
    return (struct dimensions){rows:rows_count, cols:cols_count}; //Number of rows and columns is returned
}

// Function for reading and storing data into a matrix
float** read_data(FILE *csv, struct dimensions csv_dimensions){
    int x = 0; //Row position into the data matrix

    int rows = csv_dimensions.rows; //Number of rows
    int cols = csv_dimensions.cols; //Number of columns

    float** data = get_matrix(rows, cols); //Creating a new matrix of (rows x cols) size

    char* token = NULL; //It stores the column value
    char buff[BUFFER_SIZE]; //It stores the lines as a string
    if (csv != NULL)
    {
        while (fgets(buff, BUFFER_SIZE, csv) != NULL && x < rows) //Iterates line by line over de csv file
        {
            int y = 0; //It indicates the columns position
            for (token = strtok(buff, ","); token != NULL && y < cols; token = strtok(NULL, ",")) //Iterates token by token over the current line
            {
                data[x][y++] = atof(token); //Stores the token into the data matrix as a float and goes to the next column
            }
            x++; //Goes to the next row or line
        }
        fclose(csv); //Closes the csv file when finished
    }
    
    return data; //Returns the data matrix
}


int main(int argc, char** argv)
{
    srand(time(NULL));//Setting seed for random numbers generation
    
    //Opening csv file
    char* fn;
    fn = argv[1];
    FILE *csv_file;
    csv_file = fopen(fn,"r");
    if(csv_file == NULL) {
        printf("Error: can't open testing data file \n");
        return -1;
    }

    struct dimensions csv_dim = get_csv_dimensions(csv_file); //Getting data dimensions from csv file
    float** data = read_data(csv_file, csv_dim); //Reading and storing data
    int rows = csv_dim.rows; //Getting rows number
    int cols = csv_dim.cols; //Getting columns number
    
    float test_ratio = 0.2; //Proportion of test data
    float train_ratio = 1.0 - test_ratio; //Proportion of training data
    int train_rows = (int)((float)rows * train_ratio); //Total number of rows for training data
    int test_rows = (int)((float)rows * test_ratio); //Total number of rows for test data
    float** dataset = shuffle(data, rows, cols); //Shuffled dataset to make partition into training and test data
    float** training_data = get_matrix(train_rows, cols); //Creating a matrix for training data
    float** testing_data = get_matrix(test_rows, cols); //Creating test data matrix
    
    //Making partition of shuffled dataset into training and test data
    for (int i = 0; i < train_rows; i++)
    {
        training_data[i] = dataset[i];
    }    
    for (int i = 0; i < test_rows; i++)
    {
        testing_data[i] = dataset[i+train_rows];
    }

    //Setting parameters for Random Forest model
    struct RF_params params = {n_estimators:10, max_depth:7, min_samples_leaf:3, max_features:3, sampling_ratio:0.9};

    float* actual = get_class_labels(testing_data, test_rows, cols); //Getting truth labels for test data
    
    // start clock for timing
    clock_t begin = clock();
    
    struct Node** rf = fit_model(training_data, params, train_rows, cols); //Fitting Random Forest model with training data
    float* predictions = get_predictions(testing_data, test_rows, rf, params.n_estimators); //Making predictions for test data
    double accuracy = get_accuracy(test_rows, actual, predictions); //Accuracy calculation from model predictions and truth labels

    // end clock for timing
    clock_t end = clock();
    printf("\ntime taken: %fs | accuracy: %.20f\n", (double)(end - begin) / CLOCKS_PER_SEC, accuracy); //Printing time taken and accuracy reached
    
    free(actual);
    free(predictions);
    free_matrix(training_data, train_rows);
    free_matrix(testing_data, test_rows);

    return 0;
}
