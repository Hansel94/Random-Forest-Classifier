#ifndef TRAININGFOREST_H
#define TRAININGFOREST_H

#include "Utils.h"

//Struct for accessing the length of an array, when splitting the dataset
struct var_array
{
    int length;
    float** array;
};

//Struct for accessing the class set and its length
struct class_label_struct
{
    int count;
    float* class_labels;
};

//Struct for splitting the dataset
struct split_params_struct
{
    int index;
    float value;
    float gini;
    struct var_array* two_halves;
};

//Struct for random forest parameters
struct RF_params
{
    int n_estimators;       // number of trees in a forest
    int max_depth;          // maximum depth of a tree 
    int min_samples_leaf;   // minimum number of data samples at a leaf node
    int max_features;       // number of features considered when calculating the best split
    float sampling_ratio;     // percentage of dataset used to fit a single decision tree at training time
};

//Struct for nodes, trees and forest
struct Node
{
    // for the split_params_struct
    int index;
    float value;
    struct var_array* two_halves;

    // child nodes if the node is an internal node
    struct Node* left;
    struct Node* right;

    // if the node is a leaf
    float left_leaf;
    float right_leaf;
};


float** subsample(float** data, float ratio, int rows, int cols);
float** shuffle(float** data, int rows, int cols);
void grow(struct Node* decision_tree, int max_depth, int min_samples_leaf, int max_features, int depth, int rows, int cols);


struct Node* create_node();
struct Node* build_tree(float** training_data, int max_depth, int min_samples_leaf, int max_features, int rows, int cols);
struct Node** fit_model(float** training_data, struct RF_params params, int rows, int cols);

struct split_params_struct best_data_split(float** data, int max_features, int rows, int cols);
struct class_label_struct get_class_values(float** data, int n, int m);
struct var_array* split_dataset(int index, float value, float** data, int rows, int cols);

float gini_index(struct var_array* two_halves, float* class_labels, int class_labels_count, int cols);

float get_leaf_node_class(float** group, int size, int cols);

#endif
