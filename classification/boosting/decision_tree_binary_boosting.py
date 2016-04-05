# Imports
import collections
import numpy as np
from math import log
from math import exp

# Function to compute the weighted mistakes
def intermediate_node_weighted_mistakes(labels_in_node, data_weights):
    """
    Purpose: Compute the weighted mistake in a decision tree node
    Input  : Labels in the node
    Output : Weights of the data points
    """
    total_weight = float(np.sum(data_weights))
    if len(data_weights) > 0:
        total_weight_positive = np.sum(data_weights[labels_in_node == +1])
    else:
        total_weight_positive =0.0
    if len(data_weights) > 0:    
        total_weight_negative = np.sum(data_weights[labels_in_node == -1])
    else:
        total_weight_negative = 0.0
    try:
        weighted_mistakes_all_positive = total_weight_negative/total_weight
    except ZeroDivisionError:
        weighted_mistakes_all_positive = total_weight_negative/(total_weight + 1.0e-15)
    try:
        weighted_mistakes_all_negative = total_weight_positive/total_weight
    except ZeroDivisionError:
        weighted_mistakes_all_negative = total_weight_positive/(total_weight + 1.0e-15)

    if weighted_mistakes_all_positive <= weighted_mistakes_all_negative:
        return (total_weight_negative, +1)
    else:
        return (total_weight_positive, -1)

# Function to identify the best feature to split
def best_splitting_feature(data, features, target, data_weights):
    """
    Purpose: Determine the best feature to split
    Input  : Data array, list of feature names, output array and data weights
    Output : Feature name
    """
    total_weight = np.sum(data_weights)
    best_error = float('+inf')
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
            left_split_weight = []
        else:
            left_split_label = target[left_split_idx]
            left_split_weight = data_weights[left_split_idx]
        
        try:
            right_split_idx = (feature_data == feature_data_key[1])
        except IndexError:
            right_split_label = []
            right_split_weight = []
        else:
            right_split_label = target[right_split_idx]
            right_split_weight = data_weights[right_split_idx]
        
        left_total_weight_mistakes, left_class = intermediate_node_weighted_mistakes(left_split_label, left_split_weight)
        right_total_weight_mistakes, right_class = intermediate_node_weighted_mistakes(right_split_label, right_split_weight)
        error = (left_total_weight_mistakes + right_total_weight_mistakes)/total_weight
        if error < best_error:
            best_error = error
            best_feature = feature_name

    return best_feature

# Function to create a leaf in a decision tree
def create_leaf(target_values, data_weights):    
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
           
    weighted_error, best_class = intermediate_node_weighted_mistakes(target_values, data_weights)
    leaf['prediction'] = best_class
    return leaf

# Function to fit data to a decision tree
def weighted_decision_tree_create(data, features, target, data_weights, current_depth = 0, max_depth = 10):
    """
    Purpose: Construct a decision tree
    Input  : Data array, list of feature names, output array, current depth of tree, max depth of tree
    Output : Decision tree
    """
    # Make a copy of the list of feature names
    remaining_features = features[:]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target))

    # Stopping condition 1
    # Check if there are mistakes at current node
    if intermediate_node_weighted_mistakes(target, data_weights) <= 1.0e-15:
        print "Stopping condition 1 reached."     
        # If there are no mistakes at current node
        # make current node a leaf node
        return create_leaf(target, data_weights)
    
    # Stopping condition 2
    # Check if there are remaining features to consider splitting on
    if len(remaining_features) == 0:
        print "Stopping condition 2 reached."    
        # If there are no remaining features to consider
        # make current node a leaf node
        return create_leaf(target, data_weights)    
    
    # Early stopping (limit tree depth)
    if current_depth >= max_depth:
        print "Reached maximum depth. Stopping for now."
        # If the max tree depth has been reached
        # make current node a leaf node
        return create_leaf(target, data_weights)
    
    # Get best feature to split on for the given data and target
    best_feature = best_splitting_feature(data, features, target, data_weights)
    # Obtain the index of the feature from the list of feature names
    feature_idx = features.index(best_feature)
    # Left split
    left_split_idx = (data[:, feature_idx] == 0)
    left_split = data[left_split_idx]
    left_split_label = target[left_split_idx]
    left_split_weight = data_weights[left_split_idx]
    # Right split
    right_split_idx = (data[:, feature_idx] == 1)
    right_split = data[right_split_idx]
    right_split_label = target[right_split_idx]
    right_split_weight = data_weights[right_split_idx]
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node."
        return create_leaf(left_split_label, left_split_weight)
    if len(right_split) == len(data):
        print "Creating leaf node."
        return create_leaf(right_split_label, right_split_weight)
    
    # Remove currently used feature from list of feature names
    remaining_features.remove(best_feature)
    print "Split on feature %s. (%s, %s)" % (best_feature, len(left_split), len(right_split))
    
    # Remove feature values from the left and right splits
    # to keep the feature name list and the data arrays consistent
    # with respect to the feature name index
    left_split = np.delete(left_split, feature_idx, 1)
    right_split = np.delete(right_split, feature_idx, 1)
    
    # Recurse on left and right subtrees
    left_tree = weighted_decision_tree_create(left_split, remaining_features, left_split_label, left_split_weight, current_depth + 1, max_depth)
    right_tree = weighted_decision_tree_create(right_split, remaining_features, right_split_label, right_split_weight, current_depth + 1, max_depth)
    
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

# Function to perform adaboost learning with decision tree stumps
def adaboost_with_tree_stumps(data, features, target, num_tree_stumps):
    """
    Purpose: Perform adaboost learning with decision trees
    Input  : Data, list of feature names, output array, number of boosting stages
    Output : Weights of boosted trees, decision trees
    """
    alpha = np.array([1.0] * len(data))
    weights = []
    tree_stumps = []
    
    for t in xrange(num_tree_stumps):
        print '====================================================='
        print 'Adaboost Iteration %d' % t
        print '====================================================='        
        # Learn a weighted decision tree stump using max_depth=1
        tree_stump = weighted_decision_tree_create(data, features, target, data_weights=alpha, max_depth=1)
        tree_stumps.append(tree_stump)
        # Make predictions
        predictions = np.apply_along_axis(lambda x: classify(tree_stump, x, features), 1, data)        
        # Produce a Boolean array indicating whether each data point was correctly classified
        is_correct = (predictions == target)
        is_wrong   = (predictions != target)        
        # Compute weighted error
        weighted_error = np.sum(alpha[is_wrong])/np.sum(alpha)         
        # Compute model coefficient using weighted error
        weight = 0.5 * log((1.0 - weighted_error)/weighted_error)
        weights.append(weight)        
        # Adjust weights on data point
        adjustment = np.where(is_correct, exp(-weight), exp(weight))        
        # Scale alpha by multiplying by adjustment and then normalize data points weights
        alpha = alpha * adjustment
        alpha = alpha/np.sum(alpha)
    
    return weights, tree_stumps

def classify_adaboost(stump_weights, tree_stumps, data, features):
    """
    Purpose: Classify a data point using an ensemble of boosted decision trees
    Input  : Weights of boosted trees, decision trees, data, list of feature names
    Output : Prediction by ensemble of decision tree
    """
    scores = np.array([0.0] * len(data))    
    for i, tree_stump in enumerate(tree_stumps):
        predictions = np.apply_along_axis(lambda x: classify(tree_stump, x, features), 1, data)
        scores = scores + predictions        
    return np.where(scores > 0.0, +1.0, -1.0)
