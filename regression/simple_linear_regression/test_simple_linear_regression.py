# Imports
import numpy as np
import pandas as pd
import simple_linear_regression as slr
import matplotlib.pyplot as plt

# Read training and test data
train_data = pd.read_csv('../data/Week01/kc_house_train_data.csv')
test_data = pd.read_csv('../data/Week01/kc_house_test_data.csv')

# Extract train input feature and output
input_feature_sqft = np.array(train_data['sqft_living'])
input_feature_bedrooms = np.array(train_data['bedrooms'])
output = np.array(train_data['price'])

# Use functions defined in simple_linear_regression.py
# to determine intercept and slope
(intercept, slope) = slr.simple_linear_regression(input_feature_sqft,
                                                  output)

# Predict for 2650 sq.ft.
predicted_output = slr.get_regression_predictions(2650,
                                                  intercept,
                                                  slope)

# RSS on train_data
train_RSS_sqft = slr.get_residual_sum_of_squares(input_feature_sqft,
                                                 output,
                                                 intercept,
                                                 slope)

# Estimated sq.ft. for $800000
estimated_sqft = slr.inverse_regression_predictions(800000,
                                                    intercept,
                                                    slope)

# Slope and intercept estimates from sqft and bedrooms features
# Train on sqft
(intercept_sqft, slope_sqft) = slr.simple_linear_regression(input_feature_sqft,
                                                            output)
# Train on bedrooms
(intercept_bedrooms, slope_bedrooms) = slr.simple_linear_regression(input_feature_bedrooms,
                                                                    output)

# Extract test input feature and output
test_input_feature_sqft = np.array(test_data['sqft_living'])
test_input_feature_bedrooms = np.array(test_data['bedrooms'])
test_output = np.array(test_data['price'])

# RSS 
test_RSS_sqft = slr.get_residual_sum_of_squares(test_input_feature_sqft,
                                                test_output,
                                                intercept,
                                                slope)
test_RSS_bedrooms = slr.get_residual_sum_of_squares(test_input_feature_bedrooms,
                                                    test_output,
                                                    intercept,
                                                    slope)

# Plots of output vs input and plots of residuals
# Price vs Sq.ft. - train data
plt.plot(input_feature_sqft, output, 'b.', label='train data')
plt.title('Price vs Sq.ft.')
plt.ylabel('Price')
plt.xlabel('Sq.ft.')
# Regression line
z = slr.get_regression_predictions(input_feature_sqft,
                                   intercept_sqft,
                                   slope_sqft)
plt.plot(input_feature_sqft, z, 'r', linewidth=2.0, label='regression line')
plt.legend(loc='upper left')
plt.show()

# Price vs Sq.ft. - Residuals
plt.plot(input_feature_sqft, output - z, '.')
plt.title('Residual - Price vs Sq.ft.')
plt.ylabel('Residual (Price - predicted price)')
plt.xlabel('Sq.ft.')
plt.show()


# Price vs Bedrooms - train data
plt.plot(input_feature_bedrooms, output, 'b.', label='train data')
plt.title('Price vs Bedrooms')
plt.ylabel('Price')
plt.xlabel('Bedrooms')
# Regression line
z = slr.get_regression_predictions(input_feature_bedrooms,
                                   intercept_bedrooms,
                                   slope_bedrooms)
plt.plot(input_feature_bedrooms, z, 'r', linewidth=2.0, label='regression line')
plt.legend(loc='upper left')
plt.show()

# Price vs Bedrooms - Residuals
plt.plot(input_feature_bedrooms, output - z, '.')
plt.title('Residual - Price vs Bedrooms')
plt.ylabel('Residual (Price - predicted price)')
plt.xlabel('Bedrooms')
plt.show()
