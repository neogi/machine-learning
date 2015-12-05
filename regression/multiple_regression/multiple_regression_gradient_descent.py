"""
Multiple Linear Regression: y = w0*x0 + w1*x1 + w2*x2 + ... + wD*xD
Objective: Estimate w0, w1, w2 ... wD given x1, x2 ... xD and y        
where xN = input feature, x0 = 1.0
      y  = output
"""

# Imports
import numpy as np

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
    return(features_matrix, output_array)

def predict(feature_matrix, weights):
    """
    Purpose: Compute predictions by multiplying the feature matrix with
             the estimated weights
    Input  : Feature matrix array, weight vector
    Output : Product of feature matrix array and weight vector
    """
    predictions = feature_matrix.dot(weights)
    return(predictions)

def fit(feature_matrix, output, initial_weights, step_size, tolerance):
    """
    Purpose: Estimate the weight vector by gradient descent
    Input  : Feature matrix array, initial weight vector, output vector,
             step size value and tolerance value
    Output : Estimated weight vector
    """
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        prediction = predict(feature_matrix, weights)
        errors = output - prediction
        gradient = -2.0 * (feature_matrix.T).dot(errors)
        gradient_magnitude = np.sqrt(np.sum(gradient * gradient))
        weights = weights - step_size * gradient
        if gradient_magnitude < tolerance:
            converged = True
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
