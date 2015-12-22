"""
LASSO Regression: y = w0*x0 + w1*x1 + w2*x2 + ... + wD*xD
Objective: Estimate w0, w1, w2 ... wD given x1, x2 ... xD and y
where xN = input feature, x0 = 1.0
      y  = output
subject to L1 penalty
"""

# Imports
import numpy as np
import copy

# Functions
def get_data(data_frame, features, output):
    """
    Purpose: Extract features and prepare a feature matrix
             Set the first feature x0 = 1
    Input  : Original Dataframe, list of feature variables, output variable
    Output : Feature matrix array, output array
    """
    data_frame['constant'] = 1.0
    features = ['constant'] + features
    features_matrix = np.array(data_frame[features])
    if output != None:    
        output_array = np.array(data_frame[output])
    else:
        output_array = []   
    return(features_matrix, output_array.reshape((len(output_array))))

def normalize(features):
    """
    Purpose: Normalize feature matrix, each column of the matrix is a feature
    Input  : Unnormalized feature matrix
    Output : Normalized feature matrix, feature norms
    """
    norms = np.linalg.norm(features, axis=0)
    normalized_features = features/norms
    return (normalized_features, norms)

def predict(feature_matrix, weights):
    """
    Purpose: Compute predictions by multiplying the feature matrix with
             the estimated weights
    Input  : Feature matrix array, weight vector
    Output : Product of feature matrix array and weight vector
    """
    predictions = feature_matrix.dot(weights)
    return(predictions)

def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    """
    Purpose: Compute the descent step for one feature
    Input  : Feature index, normalized feature matrix, output,
             feature weights and L1_penalty
    Output : Descent step for feature
    """
    predictions = feature_matrix.dot(weights)
    rho = (feature_matrix[:, i].T).dot(output - predictions + (weights[i] * feature_matrix[:, i]))
    if i==0:
        new_weight = rho
    elif rho < (-l1_penalty/2.0):
        new_weight = rho + (l1_penalty/2.0)
    elif rho > (l1_penalty/2.0):
        new_weight = rho - (l1_penalty/2.0)
    else:
        new_weight = 0.0
    return new_weight

def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    """
    Purpose: Perform cyclical coordinate descent
    Input  : Normalized feature matrix, output, initial weights,
             L1_penalty and tolerance for stopping the process
    Output : Final weights after the convergence of the coordinate
             descent procedure
    """
    D = feature_matrix.shape[1]
    weights = copy.copy(initial_weights)
    change = np.zeros(initial_weights.shape)
    converged = False
    
    while not converged:
        # Evaluate over all features
        for idx in range(D):
            # New weight for feature
            new_weight = lasso_coordinate_descent_step(idx, feature_matrix,
                                                       output, weights,
                                                       l1_penalty)
            # Compute change in weight for feature
            change[idx] = np.abs(new_weight - weights[idx])
            # assign new weight
            weights[idx] = new_weight
        # Maximum change in weight, after all changes have been computed
        max_change = max(change)
        if max_change < tolerance:
            converged = True
    return weights

def fit(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    """
    Purpose: Wrapper for cyclical coordinate descent function
    Input  : Feature matrix array, initial weight vector, output vector,
             tolerance value, l1_penalty
    Output : Estimated weight vector
    """
    weights = lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance)
    return(weights)

def get_residual_sum_of_squares(feature_matrix, weights, output):
    """
    Purpose: Compute Residual Sum of Squares (RSS)
    Input  : Feature matrix, weight vector, output vector
    Output : Residual sum of squares = sum((actual output (y) - predicted output)^2)
    """
    predictions = predict(feature_matrix, weights)
    residual = np.sum((predictions - output) ** 2)
    return(residual)
