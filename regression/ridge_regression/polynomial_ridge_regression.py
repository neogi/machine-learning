# Imports
import pandas as pd
import numpy as np

# Functions
def polynomial_dataframe(feature, degree):
    """
    Purpose: Compute higher degrees of a numeric featura
    Input  : feature, maximum degree 
    Output : Dataframe with original feature followed by
             higher degrees of feature
    """
    poly_dataframe = pd.DataFrame()
    poly_dataframe['power_1'] = feature
    if degree > 1:
        for power in range(2, degree+1):
            name = 'power_' + str(power)
            poly_dataframe[name] = feature ** power
    return poly_dataframe

def get_residual_sum_of_squares(model, features, output):
    """
    Purpose: Compute Residual Sum of Squares (RSS)
    Input  : features (x), output (y) and estimated model 
    Output : Residual sum of squares = sum((actual output (y) - predicted output)^2)
    """
    RSS = np.sum((output - model.predict(features)) ** 2)
    return(RSS)

def k_fold_cross_validation(k, l2_penalty, data, output, model):
    """
    Purpose: Perform k-fold cross validation
    Input  : Number of folds (k), L2 penalty value (l2_penalty),
             features (x), output (y), model to fit
    Output : 
    """
    n = len(data)
    validation_error = 0.0
    
    for i in xrange(k):
        start = (n*i)/k
        end = (n*(i+1))/k-1
        train_shuffled_data = data[0:start].append(data[end+1:n])
        output_train_shuffled_data = output[0:start].append(output[end+1:n])
        
        valid_shuffled_data = data[start:end+1]
        output_valid_shuffled_data = output[start:end+1]
        
        model.fit(train_shuffled_data, output_train_shuffled_data)
        validation_error = get_residual_sum_of_squares(model, valid_shuffled_data, output_valid_shuffled_data)
        
    average_validation_error = validation_error/k    
    return average_validation_error
