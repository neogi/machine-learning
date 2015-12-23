# Imports
import pandas as pd
import numpy as np
import ridge_regression_gradient_descent as rrgd
import matplotlib.pyplot as plt

# Read training and test data
dtype_dict = {'bathrooms':float, 'waterfront':int,
              'sqft_above':int, 'sqft_living15':float,
              'grade':int, 'yr_renovated':int,
              'price':float, 'bedrooms':float,
              'zipcode':str, 'long':float,
              'sqft_lot15':float, 'sqft_living':float,
              'floors':str, 'condition':int,
              'lat':float, 'date':str,
              'sqft_basement':int, 'yr_built':int,
              'id':str, 'sqft_lot':int, 'view':int}

# Read training and test data
train_data = pd.read_csv('../data/Week04/kc_house_train_data.csv', dtype=dtype_dict)
test_data = pd.read_csv('../data/Week04/kc_house_test_data.csv', dtype=dtype_dict)

# Simple model - one feature
features = ['sqft_living']
target = ['price']
init_weights = np.array([0.0, 0.0]).reshape((2, 1))
iterations = 1000
step = 1e-12
tolerance = 0

(train_feature_matrix, train_output) = rrgd.get_data(train_data, features, target)
(test_feature_matrix, test_output) = rrgd.get_data(test_data, features, target)

l2_penalty = 0
simple_model = rrgd.fit(train_feature_matrix,
                        train_output,
                        init_weights,
                        step_size=step,
                        l2_penalty=l2_penalty,
                        tolerance=tolerance,
                        max_iterations=iterations)

l2_penalty = 1e11
simple_model_l2 = rrgd.fit(train_feature_matrix,
                        train_output,
                        init_weights,
                        step_size=step,
                        l2_penalty=l2_penalty,
                        tolerance=tolerance,
                        max_iterations=iterations)

plt.title('L2 penalty comparison')
plt.ylabel('Price')
plt.xlabel('Sq.ft.')
plt.plot(train_feature_matrix[:,1], train_output, 'k.', label='training data')
plt.plot(train_feature_matrix[:,1], rrgd.predict(train_feature_matrix, simple_model), 'b-', label='L2=0')
plt.plot(train_feature_matrix[:,1], rrgd.predict(train_feature_matrix, simple_model_l2), 'r-', label='L2=1e11')
plt.legend(loc='upper left')
plt.show()

simple_model_test_RSS = rrgd.get_residual_sum_of_squares(test_feature_matrix,
                                                         init_weights,
                                                         test_output)

simple_model_test_RSS_noL2 = rrgd.get_residual_sum_of_squares(test_feature_matrix,
                                                         simple_model,
                                                         test_output)

simple_model_test_RSS_L2 = rrgd.get_residual_sum_of_squares(test_feature_matrix,
                                                         simple_model_l2,
                                                         test_output)

# Complex model - two features
features = ['sqft_living', 'sqft_living15']
target = ['price']
init_weights = np.array([0.0, 0.0, 0.0]).reshape((3, 1))
iterations = 1000
step = 1e-12
tolerance = 0

(train_feature_matrix, train_output) = rrgd.get_data(train_data, features, target)
(test_feature_matrix, test_output) = rrgd.get_data(test_data, features, target)

l2_penalty = 0
complex_model = rrgd.fit(train_feature_matrix,
                        train_output,
                        init_weights,
                        step_size=step,
                        l2_penalty=l2_penalty,
                        tolerance=tolerance,
                        max_iterations=iterations)

l2_penalty = 1e11
complex_model_l2 = rrgd.fit(train_feature_matrix,
                        train_output,
                        init_weights,
                        step_size=step,
                        l2_penalty=l2_penalty,
                        tolerance=tolerance,
                        max_iterations=iterations)

complex_model_test_RSS = rrgd.get_residual_sum_of_squares(test_feature_matrix,
                                                         init_weights,
                                                         test_output)

complex_model_test_RSS_noL2 = rrgd.get_residual_sum_of_squares(test_feature_matrix,
                                                         complex_model,
                                                         test_output)

complex_model_test_RSS_L2 = rrgd.get_residual_sum_of_squares(test_feature_matrix,
                                                         complex_model_l2,
                                                         test_output)
