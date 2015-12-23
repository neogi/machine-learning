# Imports
import numpy as np
import pandas as pd
import multiple_regression_gradient_descent as mrgd

# Read training and test data
train_data = pd.read_csv('../data/Week02/kc_house_train_data.csv')
test_data = pd.read_csv('../data/Week02/kc_house_test_data.csv')

# Model 1
# Prepare train and test data for gradient descent method
features = ['sqft_living']
output = ['price']
(feature_matrix1, output_vector) = mrgd.get_data(train_data, features, output)
(feature_matrix1_t, output_vector_t) = mrgd.get_data(test_data, features, output)

# Set parameters
step_size1 = 7.0e-12
tolerance1 = 2.5e7
init_weights1 = np.array([-47000.0, 1.0]).reshape((2, 1))

# Train on train data
model1_weights = mrgd.fit(feature_matrix1,
                          output_vector,
                          init_weights1,
                          step_size1,
                          tolerance1)

# Predict on test data
test1_predictions = mrgd.predict(feature_matrix1_t, model1_weights)
RSS1 = mrgd.get_residual_sum_of_squares(feature_matrix1_t, model1_weights, output_vector_t)

# Model 2
# Prepare train and test data for gradient descent method
features = ['sqft_living', 'sqft_living15']
output = ['price']
(feature_matrix2, output_vector) = mrgd.get_data(train_data, features, output)
(feature_matrix2_t, output_vector_t) = mrgd.get_data(test_data, features, output)

# Set parameters
step_size2 = 4.0e-12
tolerance2 = 1.0e9
init_weights2 = np.array([-100000.0, 1.0, 1.0]).reshape((3, 1))

# Train on train data
model2_weights = mrgd.fit(feature_matrix2,
                          output_vector,
                          init_weights2,
                          step_size2,
                          tolerance2)

# Predict on test data
test2_predictions = mrgd.predict(feature_matrix2_t, model2_weights)
RSS2 = mrgd.get_residual_sum_of_squares(feature_matrix2_t, model2_weights, output_vector_t)
