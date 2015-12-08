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
