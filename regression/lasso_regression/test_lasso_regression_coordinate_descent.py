# Imports
import numpy as np
import pandas as pd
import lasso_regression_coordinate_descent as lasso
from math import sqrt

# Test Lasso step
print 'Step: ' + str(
       lasso.lasso_coordinate_descent_step(1,
                                          np.array([[3./sqrt(13),1./sqrt(10)],
                                                    [2./sqrt(13),3./sqrt(10)]]),
                                          np.array([1., 1.]),
                                          np.array([1., 4.]),
                                          0.1))
print 'Expected: 0.425558846691'
print ''

# Test convergence on a simple data set
#features = np.array([[3./sqrt(13), 1./sqrt(10)], [2./sqrt(13),3./sqrt(10)]])
#output = np.array([1., 1.])
#init_weights = np.array([1., 4.])
#l1_penalty = 0.1
#tolerance = 0.1
#model = lasso.fit(features, output, init_weights, l1_penalty, tolerance)

# Data type for house sales data
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
              'sqft_living15':float, 'grade':int, 'yr_renovated':int,
              'price':float, 'bedrooms':float, 'zipcode':str,
              'long':float, 'sqft_lot15':float, 'sqft_living':float,
              'floors':float, 'condition':int, 'lat':float, 'date':str,
              'sqft_basement':int, 'yr_built':int, 'id':str,
              'sqft_lot':int, 'view':int}

# Read house sales data
sales = pd.read_csv('../data/Week05/kc_house_data.csv', dtype=dtype_dict)
# Select features and ouput
features = ['sqft_living', 'bedrooms']
target = ['price']
# Extract and normalize feature matrix
feature_matrix, output = lasso.get_data(sales, features, target)
norm_feature_matrix, norms = lasso.normalize(feature_matrix)

# Model for all sales data
init_weights = np.zeros((len(features)+1))
l1_penalty = 1.0e7
tolerance = 1.0

model = lasso.fit(norm_feature_matrix, output, init_weights, l1_penalty, tolerance)
RSS = lasso.get_residual_sum_of_squares(norm_feature_matrix, model, output)

# Read house sales train and test data
train_data = pd.read_csv('../data/Week05/kc_house_train_data.csv', dtype=dtype_dict)
test_data = pd.read_csv('../data/Week05/kc_house_test_data.csv', dtype=dtype_dict)
# Select features and output
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
            'floors', 'waterfront', 'view', 'condition', 'grade',
            'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
target = ['price']
# Extract train and test feature matrix, normalize only the train feature matrix
train_feature_matrix, train_output = lasso.get_data(train_data, features, target)
norm_train_feature_matrix, norm_train = lasso.normalize(train_feature_matrix)
test_feature_matrix, test_output = lasso.get_data(train_data, features, target)

# Model 1 for train data
init_weights = np.zeros((len(features)+1))
l1_penalty = 1.0e7
tolerance = 1.0
weights1e7 = lasso.fit(norm_train_feature_matrix, train_output, init_weights, l1_penalty, tolerance)

# Model 2 for train data
init_weights = np.zeros((len(features)+1))
l1_penalty = 1.0e8
tolerance = 1.0
weights1e8 = lasso.fit(norm_train_feature_matrix, train_output, init_weights, l1_penalty, tolerance)

# Model 3 for train data
init_weights = np.zeros((len(features)+1))
l1_penalty = 1.0e4
tolerance = 5.0e5
weights1e4 = lasso.fit(norm_train_feature_matrix, train_output, init_weights, l1_penalty, tolerance)

# Normalize the models using the norms of the train data
norm_weights1e7 = weights1e7/norm_train
norm_weights1e8 = weights1e8/norm_train
norm_weights1e4 = weights1e4/norm_train

# Compute RSS on test data using normalized weights and unnormalized test data
RSS_1e7 = lasso.get_residual_sum_of_squares(test_feature_matrix, norm_weights1e7, test_output)
RSS_1e8 = lasso.get_residual_sum_of_squares(test_feature_matrix, norm_weights1e8, test_output)
RSS_1e4 = lasso.get_residual_sum_of_squares(test_feature_matrix, norm_weights1e4, test_output)
