# Imports
import collections
import numpy as np

# Function to compute the number of mistakes made by majority classification
def intermediate_node_num_mistakes(labels_in_node):
    """
    Purpose: Compute the number of mistakes in a decision tree node
             using majority classification
    Input  : Labels in the node
    Output : Number of mistakes
    """
    if len(labels_in_node) == 0:
        number_of_mistakes = 0
    else:
        class_counts = collections.Counter(labels_in_node)
        class_counts_positive = class_counts[1]
        class_counts_negative = class_counts[-1]
        if class_counts_positive >= class_counts_negative:
            number_of_mistakes = class_counts_negative
        else:
            number_of_mistakes = class_counts_positive
    return number_of_mistakes

# Function to identify the best feature to split
def best_splitting_feature(data, features, target):
    """
    Purpose: Determine the best feature to split
    Input  : Data array, list of feature names and output array
    Output : Feature name
    """
    number_of_data_points = float(len(data))
    best_error = 10.0
    best_feature = None
    
    for feature_name in features:
        feature_idx = features.index(feature_name)
        feature_data = data[:, feature_idx]
        feature_data_groups = collections.Counter(feature_data)
        feature_data_key = feature_data_groups.keys()
        
        try:
            left_split_idx = (feature_data == feature_data_key[0])
        except IndexError:
            left_split_label = []
        else:
            left_split_label = target[left_split_idx]
        
        try:
            right_split_idx = (feature_data == feature_data_key[1])
        except IndexError:
            right_split_label = []
        else:
            right_split_label = target[right_split_idx]
        
        left_mistakes = intermediate_node_num_mistakes(left_split_label)
        right_mistakes = intermediate_node_num_mistakes(right_split_label)
        error = (left_mistakes + right_mistakes)/number_of_data_points
        if error < best_error:
            best_error = error
            best_feature = feature_name
        
    return best_feature

# Function to create a leaf in a decision tree
def create_leaf(target_values):    
    """
    Purpose: Create a leaf node
    Input  : Output labels
    Output : Leaf node with label based on majority classification
    """
    leaf = {
                'splitting_feature': None,
                'left':              None,
                'right':             None,
                'is_leaf':           True
           }
           
    class_counts_positives = (len(target_values[target_values == +1]))
    class_counts_negatives = (len(target_values[target_values == -1]))

    if class_counts_positives > class_counts_negatives:
        leaf['prediction'] = +1
    else:
        leaf['prediction'] = -1
    return leaf


# Function to check if the minimum node size has been reached
def reached_minimum_node_size(data, min_node_size):
    """
    Purpose: Determine if the node contains at most the minimum
             number of data points
    Input  : Data array and minimum node size
    Output : True if minimum size has been reached, else False
    """
    if len(data) <= min_node_size:
        return True
    else:
        return False


# Function to compute the error difference before and after splitting
def error_reduction(error_before_split, error_after_split):
    """
    Purpose: Compute the difference between error before and after split
    Input  : Error before split and error after split
    Output : Difference between error before and after split
    """
    return (error_before_split - error_after_split)


# Function to fit data to a decision tree
def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10, 
                         min_node_size = 1, min_error_reduction = 0.0):
    """
    Purpose: Construct a decision tree
    Input  : Data array, list of feature names, output array,
             current depth of tree, max depth of tree, minimum node size
             and minimum error reduction
    Output : Decision tree
    """
    # Make a copy of the list of feature names
    remaining_features = features[:]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target))

    # Stopping condition 1
    # Check if there are mistakes at current node
    if intermediate_node_num_mistakes(target) == 0:
        print "Stopping condition 1 reached."     
        # If there are no mistakes at current node
        # make current node a leaf node
        return create_leaf(target)
    
    # Stopping condition 2
    # Check if there are remaining features to consider splitting on
    if len(remaining_features) == 0:
        print "Stopping condition 2 reached."    
        # If there are no remaining features to consider
        # make current node a leaf node
        return create_leaf(target)    
    
    # Early stopping (limit tree depth)
    if current_depth >= max_depth:
        print "Reached maximum depth. Stopping for now."
        # If the max tree depth has been reached
        # make current node a leaf node
        return create_leaf(target)
    
    # Early stopping (minimum node size)
    if len(data) <= min_node_size:
        print "Reached minimum node size."
        # If the number of data point are sufficient
        # make current node a leaf node
        return create_leaf(target)
    
    # Get best feature to split on for the given data and target
    best_feature = best_splitting_feature(data, features, target)
    # Obtain the index of the feature from the list of feature names
    feature_idx = features.index(best_feature)
    # Left split
    left_split_idx = (data[:, feature_idx] == 0)
    left_split = data[left_split_idx]
    left_split_label = target[left_split_idx]
    # Right split
    right_split_idx = (data[:, feature_idx] == 1)
    right_split = data[right_split_idx]
    right_split_label = target[right_split_idx]
    
    # Early stopping (minimum error reduction)
    error_before_split = intermediate_node_num_mistakes(target)/float(len(data))
    left_split_error = intermediate_node_num_mistakes(left_split_label)
    right_split_error = intermediate_node_num_mistakes(right_split_label)
    error_after_split = (left_split_error + right_split_error)/float(len(data))
    if error_reduction(error_before_split, error_after_split) <= min_error_reduction:
        print "Reached minimum error reduction limit."
        # If the split does not achieve the minimum error reduction
        # make current node a leaf node
        return create_leaf(target)
    
    # This section of code was used when the early stopping criteria were not implemented
    # Create a leaf node if the split is "perfect"
    #if len(left_split) == len(data):
    #    print "Creating leaf node."
    #    return create_leaf(left_split_label)
    #if len(right_split) == len(data):
    #    print "Creating leaf node."
    #    return create_leaf(right_split_label)
    
    # Remove currently used feature from list of feature names
    remaining_features.remove(best_feature)
    print "Split on feature %s. (%s, %s)" % (best_feature, len(left_split), len(right_split))
    
    # Remove feature values from the left and right splits
    # to keep the feature name list and the data arrays consistent
    # with respect to the feature name index
    left_split = np.delete(left_split, feature_idx, 1)
    right_split = np.delete(right_split, feature_idx, 1)
    
    # Recurse on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, left_split_label, current_depth + 1, max_depth, min_node_size, min_error_reduction)
    right_tree = decision_tree_create(right_split, remaining_features, right_split_label, current_depth + 1, max_depth, min_node_size, min_error_reduction)
    
    return {
                'is_leaf'          : False, 
                'prediction'       : None,
                'splitting_feature': best_feature,
                'left'             : left_tree, 
                'right'            : right_tree
           }

# Function to predict classes given a decision tree and data
def classify(tree, x, features, annotate = False):
    """
    Purpose: Classify a data point using a decision tree
    Input  : Decision tree, data point, list of feature names
             option to annotate the path taken by the decision tree
    Output : Prediction by decision tree
    """
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction']
    else:
        # split on feature.
        feature = tree['splitting_feature']
        feature_idx = features.index(feature)
        feature_value = x[feature_idx]
        if annotate:
            print "Split on %s = %s" % (tree['splitting_feature'], feature_value)
        if feature_value == 0:
            return classify(tree['left'], x, features, annotate)
        else:
            return classify(tree['right'], x, features, annotate)

# Function to count the number of leaves in a tree
def count_leaves(tree):
    """
    Purpose: Count the number of leaf nodes in a decision tree
    Input  : Decision tree
    Output : The number of leaves in the decision tree
    """
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])

# Function to display a decision node in a decision tree
def print_stump(tree, name = 'root'):
    """
    Purpose: Print a decision tree node
    Input  : Decision tree, root name
    Output : Character based graph of decision tree node
    """
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('.')
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))

# Function to compute accuracy
def accuracy(prediction, actual):
    """
    Purpose: Compute accuracy
    Input  : Predicted output values, true output values
    Output : Accuracy
    """
    prediction_correct = sum((actual == prediction)*1.0)
    prediction_total = len(prediction)
    accuracy = prediction_correct/prediction_total
    return accuracy
