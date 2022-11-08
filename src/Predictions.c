#include <stdio.h>
#include <stdlib.h>

#include "Predictions.h"

//Get predictions based on evaluations per each node in a tree
float tree_prediction(struct Node *decision_tree, float* row)
{
    if(row[decision_tree->index] < decision_tree->value)
    {
        if(decision_tree->left != NULL)
        {
            return tree_prediction(decision_tree->left, row);
        }
        else
        {
            return decision_tree->left_leaf;
        }
    } else {
        if(decision_tree->right != NULL)
        {
            return tree_prediction(decision_tree->right, row);
        }
        else
        {
            return decision_tree->right_leaf;
        }
    }
}

//Determines the class for an instance based on majority votes
//Only implemented for binary classification
float majority_votes(struct Node** trees, int n_estimators, float* row)
{
    int zeroes = 0;
    int ones = 0;
    for(int i=0; i < n_estimators; i++)
    {
        float prediction = tree_prediction(trees[i], row);
        if(prediction == 0) zeroes++;
        if(prediction == 1) ones++;
    }
    if(ones > zeroes)
    {
        return 1;
    }
    else {
        return 0;
    }
}

//Get the predictions for the entire forest
float* get_predictions(float** test_data, int rows_test, struct Node** trees, int n_estimators)
{
    float* predictions = malloc(rows_test * sizeof(float));
    for(int i=0; i < rows_test; i++)
    {
        float* row = test_data[i];
        float prediction = majority_votes(trees, n_estimators, row);
        predictions[i] = prediction;
    }
    return predictions;
}


float* get_class_labels(float** training_data, int rows_test_data, int cols)
{
    float* class_labels = malloc(rows_test_data * sizeof(float));
    for(int i=0; i < rows_test_data; i++)
    {
        float* row = training_data[i];
        class_labels[i] = row[cols-1];
    }
    return class_labels;
}