"""
Ridge Regression: y = w0*x0 + w1*x1 + w2*x2 + ... + wD*xD
Objective: Estimate w0, w1, w2 ... wD given x1, x2 ... xD and y
where xN = input feature, x0 = 1.0
      y  = output
subject to L2 penalty
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

def fit(feature_matrix, output, initial_weights, step_size, l2_penalty, tolerance, max_iterations=100):
    """
    Purpose: Estimate the weight vector by gradient descent
    Input  : Feature matrix array, initial weight vector, output vector,
             step size value, tolerance value, l2_penalty and maximum iterations
    Output : Estimated weight vector
    """
    converged = False
    weights = np.array(initial_weights)
    weights_scaler = np.ones(len(weights))
    weights_scaler[1:] = 1.0 - 2.0 * step_size * l2_penalty
    weights_scaler = weights_scaler.reshape((len(weights),1))
    iteration = 0
    print 'Gradient descent'
    while not converged:
        prediction = predict(feature_matrix, weights)
        errors = output - prediction
        product = (feature_matrix.T).dot(errors)
        gradient = -2.0 * product + 2.0 * l2_penalty * weights
        gradient_magnitude = np.sqrt(np.sum(gradient * gradient))
        weights = weights_scaler * weights + 2.0 * step_size * product
        iteration += 1
        if (iteration > max_iterations) or (gradient_magnitude < tolerance):
            converged = True
            print 'Stopping at iteration: ' + str(iteration - 1)
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
