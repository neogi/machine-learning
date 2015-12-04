"""
Simple Linear Regression: y = w0 + w1*x
Objective: Estimate w0 and w1, given x and y
where x  = input feature
      y  = output
      w0 = intercept
      w1 = slope
"""

# Imports
import numpy as np

# Functions
def simple_linear_regression(input_feature, output):
    """
    Purpose: Compute intercept and slope
    Input  : input_feature (x), output (y)
    Output : Estimate of intercept (w0) and slope (w1)
    """
    mean_input_feature = np.mean(input_feature)
    mean_output = np.mean(output)
    slope = np.sum(input_feature * output - input_feature * mean_output)/np.sum(input_feature * input_feature - input_feature * mean_input_feature)
    intercept = mean_output - slope * mean_input_feature
    return(intercept, slope)

def get_regression_predictions(input_feature, intercept, slope):
    """
    Purpose: Compute predictions
    Input  : input_feature (x), intercept (w0), slope (w1)
    Output : Predicted output based on estimated intercept, slope and input feature
    """
    predicted_output = intercept + slope * input_feature
    return(predicted_output)

def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    """
    Purpose: Compute Residual Sum of Squares (RSS)
    Input  : input_feature (x), output (y), 
             intercept (w0), slope (w1)
    Output : Residual sum of squares = sum((actual output (y) - predicted output)^2)
    """
    RSS = np.sum((output - (intercept + slope * input_feature)) ** 2)
    return(RSS)

def inverse_regression_predictions(output, intercept, slope):
    """
    Purpose: Compute Residual Sum of Squares (RSS)
    Input  : output (y), intercept (w0), slope (w1)
    Output : Estimate of input based on intercept, slope and given output
    """
    estimated_input = (output - intercept)/slope
    return(estimated_input)
