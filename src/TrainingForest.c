#include <stdio.h>
#include <limits.h>

#include "TrainingForest.h"

//Node constructor
struct Node* create_node()
{
    struct Node* newNode = malloc(sizeof(struct Node)); //allocates memory for a new node
    newNode->left = NULL; //Node points to NULL at its left
    newNode->right = NULL; //Node points to NULL at its right
    newNode->two_halves = NULL;
    newNode->index = -1;
    newNode->value = -1;
    newNode->right_leaf = -1;
    newNode->left_leaf = -1;
    return newNode;
}

//Builds the entire random forest
struct Node** fit_model(float** training_data, struct RF_params params, int rows, int cols)
{

    struct Node** trees = (struct Node**) malloc(sizeof(struct Node) * params.n_estimators);
    for(int i=0; i < params.n_estimators; i++)
    {
        float** sample = subsample(training_data, params.sampling_ratio, rows, cols);
        int sample_row_count = (int)((float)rows * params.sampling_ratio);

        struct Node* tree = build_tree(sample, params.max_depth, params.min_samples_leaf, params.max_features, sample_row_count, cols);
        trees[i] = tree;
    }
    return trees;
}

//Builds a tree with specified parameters
struct Node* build_tree(float** training_data, int max_depth, int min_samples_leaf, int max_features, int rows, int cols)
{
    struct split_params_struct split_params = best_data_split(training_data, max_features, rows, cols);
    struct Node* root = create_node();
    root->two_halves = split_params.two_halves;
    root->index = split_params.index;
    root->value = split_params.value;
    grow(root, max_depth, min_samples_leaf, max_features, 1, rows, cols);
    return root;
}

//Gets the class set
struct class_label_struct get_class_values(float** data, int n, int m)
{
    int count = 0;
    // create an array to hold class values
    float* class_value_set = malloc(count * sizeof(float));

    for(int i=0; i < n; i++)
    {
        float class_target = data[i][m-1];
        if(!contains_float(class_value_set, count, class_target))
        {
            count++;
            float* temp = realloc(class_value_set, count * sizeof(float));
            if(temp != NULL) class_value_set = temp;
            class_value_set[count-1] = class_target;
        }
    }
    return (struct class_label_struct){count, class_value_set};
}

//Gets the class for leaf nodes
float get_leaf_node_class(float** group, int size, int cols)
{
    int zeroes = 0;
    int ones = 0;
    for(int i=0; i < size; i++)
    {
        float* row = group[i];
        float class_label = row[cols-1];
        if(class_label == 0) zeroes++;
        if(class_label == 1) ones++;
    }
    if(ones >= zeroes)
    {
        return 1;
    }
    else {
        return 0;
    }
}

//Grows the tree recursively
void grow(struct Node* decision_tree, int max_depth, int min_samples_leaf, int max_features, int depth, int rows, int cols)
{
    struct Node obj = *decision_tree;
    struct var_array left_half = obj.two_halves[0];
    struct var_array right_half = obj.two_halves[1];

    float** left = left_half.array;
    float** right = right_half.array;
    decision_tree->two_halves = NULL;

    if(left == NULL || right == NULL)
    {
        float leaf = get_leaf_node_class(combine_matrices(left, right, left_half.length,right_half.length, cols), rows, cols);
        decision_tree->left_leaf = leaf;
        decision_tree->right_leaf = leaf;
        return;
    }
    if(depth >= max_depth)
    {
        decision_tree->left_leaf = get_leaf_node_class(left, left_half.length, cols);
        decision_tree->right_leaf = get_leaf_node_class(right, right_half.length, cols);
        return;
    }
    if(left_half.length <= min_samples_leaf)
    {
        decision_tree->left_leaf = get_leaf_node_class(left, left_half.length, cols);
    }
    else {
        struct split_params_struct split_params = best_data_split(left, max_features, left_half.length, cols);
        decision_tree->left = create_node();
        decision_tree->left->two_halves = split_params.two_halves;
        decision_tree->left->index = split_params.index;
        decision_tree->left->value = split_params.value;
        grow(decision_tree->left, max_depth, min_samples_leaf, max_features, depth+1, rows, cols);
    }
    if(right_half.length <= min_samples_leaf)
    {
        decision_tree->right_leaf = get_leaf_node_class(right, right_half.length, cols);
    }
    else {
        struct split_params_struct split_params = best_data_split(right, max_features, right_half.length, cols);
        decision_tree->right = create_node();
        decision_tree->right->two_halves = split_params.two_halves;
        decision_tree->right->index = split_params.index;
        decision_tree->right->value = split_params.value;
        grow(decision_tree->right, max_depth, min_samples_leaf, max_features, depth+1, rows, cols);
    }
}

//splits the dataset depending on the given feature
struct var_array* split_dataset(int index, float value, float** data, int rows, int cols)
{
    float** left = get_matrix(1, cols);
    float** right = get_matrix(1, cols);

    int left_count = 0;
    int right_count = 0;

    for(int i = 0; i < rows; i++)
    {
        float* row = data[i];
        if(row[index] < value)
        {
            left[left_count] = row;
            left_count++;
            float** temp = realloc(left, left_count * sizeof(float) * cols);
            if (temp != NULL) left = temp;
        }
        else {
            // copy the row into right half
            right[right_count] = row;
            right_count++;
            float** temp = realloc(right, right_count * sizeof(float) * cols);
            if (temp != NULL) right = temp;
        }
    }
    struct var_array* left_right = malloc(sizeof(struct var_array) * 2);

    left_right[0] = (struct var_array){left_count, left};
    left_right[1] = (struct var_array){right_count, right};
    return left_right;
}

//Calculates gini index
float gini_index(struct var_array* two_halves, float* class_labels, int class_labels_count, int cols)
{
    int count = 2; // two halves expected
    int n_instances = two_halves[0].length + two_halves[1].length;
    float gini = 0.0;
    for(int i=0; i < count; i++)
    {
        struct var_array group = two_halves[i];
        int size = group.length;
        if(size == 0) continue;
        // else compute the score
        float sum = 0.0;

        for(int j=0; j < class_labels_count; j++)
        {
            float class = class_labels[j];
            float occurences = 0;
            for(int k=0; k < size; k++)
            {
                float label = group.array[k][cols-1];
                if(label == class)
                {
                    occurences += 1.0;
                }
            }
            float p_class = occurences / (float)size;
            sum += (p_class * p_class);
        }
        gini += (1.0 - sum) * ((float)size / (float)n_instances);
    }

    return gini;
}

//Determines the best feature to split the data based on gini index evaluation
struct split_params_struct best_data_split(float** data, int max_features, int rows, int cols)
{

    struct class_label_struct classes_struct = get_class_values(data, rows, cols);
    struct var_array* best_two_halves = NULL;

    float best_value = (float)INT_MAX;
    float best_gini = (float)INT_MAX;
    int best_index = INT_MAX;

    // create a features array and initialize to avoid non-set memory
    int* features = malloc(max_features* sizeof(int));
    for(int i=0; i < max_features; i++) features[i] = -1;

    int count = 0;
    while(count < max_features)
    {
        int max = cols-2;
        int min = 0;
        int index = rand() % (max + 1 - min) + min;
        if(!contains_int(features, max_features, index))
        {
            features[count] = index;
            count++;
        }
    }
    for(int i=0; i < max_features; i++)
    {
        int index = features[i];
        for(int j=0; j < rows; j++)
        {
            float* row = data[j];
            struct var_array* two_halves = split_dataset(index, row[index], data, rows, cols);
            float gini = gini_index(two_halves, classes_struct.class_labels, classes_struct.count, cols);

            if(gini < best_gini)
            {
                best_index = index;
                best_value = row[index];
                best_gini = gini;
                best_two_halves = two_halves;
            }
        }
    }

    free(features);
    return (struct split_params_struct){best_index, best_value, best_gini, best_two_halves};
}

//Gets a random subsample from the training dataset
float** subsample(float** data, float ratio, int rows, int cols)
{
    int sample_rows = (int)((float)rows * ratio);
    float** sample = get_matrix(sample_rows, cols);

    int* indices = malloc(sample_rows * sizeof(int));
    int count = 0;
    for(int i=0; i < sample_rows; i++) indices[i] = -1;

    while(count < sample_rows)
    {
        int max = rows-1;
        int random_index = rand() % (max + 1);
        if(!contains_int(indices, sample_rows, random_index))
        {
            sample[count] = data[random_index];

            // keep track of this index so that there are no duplicate rows
            indices[count] = random_index;
            count++;
        }
    }
    return sample;
}

//Shuffles the entire training dataset
float** shuffle(float** data, int rows, int cols)
{
    float** sample = get_matrix(rows, cols);

    int* indices = malloc(rows * sizeof(int));
    int count = 0;
    for(int i=0; i < rows; i++) indices[i] = -1;

    while(count < rows)
    {
        int max = rows-1;
        int random_index = rand() % (max + 1);
        if(!contains_int(indices, rows, random_index))
        {
            sample[count] = data[random_index];
            indices[count] = random_index;
            count++;
        }
    }
    return sample;
}
