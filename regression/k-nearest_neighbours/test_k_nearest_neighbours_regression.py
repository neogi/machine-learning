# Imports
import numpy as np
import pandas as pd
import k_nearest_neighbours_regression as knn

# Data type for house sales data
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,
              'sqft_living15':float, 'grade':int, 'yr_renovated':int,
              'price':float, 'bedrooms':float, 'zipcode':str,
              'long':float, 'sqft_lot15':float, 'sqft_living':float,
              'floors':float, 'condition':int, 'lat':float, 'date':str,
              'sqft_basement':int, 'yr_built':int, 'id':str,
              'sqft_lot':int, 'view':int}

# Read house sales data
train = pd.read_csv('../data/Week06/kc_house_data_small_train.csv', dtype=dtype_dict)
valid = pd.read_csv('../data/Week06/kc_house_data_validation.csv', dtype=dtype_dict)
test = pd.read_csv('../data/Week06/kc_house_data_small_test.csv', dtype=dtype_dict)
# Feature list
feature_list = ['bedrooms', 'bathrooms', 'sqft_living',
                'sqft_lot', 'floors', 'waterfront',  
                'view', 'condition', 'grade', 'sqft_above',  
                'sqft_basement', 'yr_built', 'yr_renovated',  
                'lat', 'long', 'sqft_living15', 'sqft_lot15']

# Extract training, validation and test data                
features_train, output_train = knn.get_data(train, feature_list, 'price')
features_test, output_test = knn.get_data(test, feature_list, 'price')
features_valid, output_valid = knn.get_data(valid, feature_list, 'price')

# Normalize training, validation and test data
norm_features_train, norms = knn.normalize(features_train)
norm_features_test = features_test / norms
norm_features_valid = features_valid / norms

# Distance between test[0] and train[9]
dist_q_test_10 = np.sqrt(np.sum((norm_features_test[0] - norm_features_train[9]) ** 2.0))
print dist_q_test_10

# Distance between test[0] and train[0:10]
dist_q_test_10 = np.sqrt(np.sum((norm_features_test[0] - norm_features_train[0:10]) ** 2.0, axis=1))
print np.argmin(dist_q_test_10) #9th house

# Distance between all train and test[2]
dist_q_test_3 = knn.compute_distances(norm_features_train, norm_features_test[2])
print np.argmin(dist_q_test_3) #383th house
print train['price'][382] # 1-NN price prediction

# 4-NN for test[2]
nearest_neighbors = knn.k_nearest_neighbors(4, norm_features_train, norm_features_test[2])
prediction = knn.predict_output_of_query(4, norm_features_train, output_train, norm_features_test[2])
print prediction

# 10-NN prediction for test[0:10]
prediction_test_10 = knn.predict(10, norm_features_train, output_train, norm_features_test[0:10])
print np.argmin(prediction_test_10), min(prediction_test_10)

# Cross validation to select best number of neighbours
RSS = np.zeros((15))
for k in range(1, 16):
    prediction_on_validation = knn.predict(k, norm_features_train, output_train, norm_features_valid)
    RSS[k-1] = knn.get_residual_sum_of_squares(prediction_on_validation, output_valid)

# Select k such that RSS is minimum    
best_k = np.argmin(RSS) + 1

# Predict on test set with best k
prediction_on_test = knn.predict(best_k, norm_features_train, output_train, norm_features_test)
RSS_test = knn.get_residual_sum_of_squares(prediction_on_test, output_test)
print RSS_test
