#ifndef PREDICTIONS_H
#define PREDICTIONS_H

#include "TrainingForest.h"

float* get_predictions(float** test_data, int rows_test, struct Node** trees, int n_estimators);
float majority_votes(struct Node** trees, int n_estimators, float* row);
float tree_prediction(struct Node *decision_tree, float* row);
float* get_class_labels(float** testing_data, int rows_test_data, int cols);

#endif