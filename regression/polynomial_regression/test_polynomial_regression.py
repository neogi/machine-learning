# Imports
import numpy as np
import pandas as pd
import polynomial_regression as plr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Function to plot data and results
def plot_fit(model, features, output):
    plt.plot(features['power_1'], output, '.', label='output')
    plt.plot(features['power_1'], model.predict(features), '-', label='regression line')
    plt.title('Price vs Sq.ft.')
    plt.ylabel('Price')
    plt.xlabel('Sq.ft.')
    plt.legend(loc='upper left')
    plt.show()

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

train_data = pd.read_csv('../data/Week03/wk3_kc_house_train_data.csv', dtype=dtype_dict)
val_data = pd.read_csv('../data/Week03/wk3_kc_house_valid_data.csv', dtype=dtype_dict)
test_data = pd.read_csv('../data/Week03/wk3_kc_house_test_data.csv', dtype=dtype_dict)

train_data = train_data.sort(['sqft_living'])
val_data = val_data.sort(['sqft_living'])
test_data = test_data.sort(['sqft_living'])

# Create order 1 polynomial feature, train and predict
poly1_data = plr.polynomial_dataframe(train_data['sqft_living'], 1)
output = train_data['price']
model1 = LinearRegression()
model1.fit(poly1_data, output)

# Plot order 1 fit
plot_fit(model1, poly1_data, output)

# Read other dataset
set_1 = pd.read_csv('../data/Week03/wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
set_2 = pd.read_csv('../data/Week03/wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
set_3 = pd.read_csv('../data/Week03/wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
set_4 = pd.read_csv('../data/Week03/wk3_kc_house_set_4_data.csv', dtype=dtype_dict)
set_1 = set_1.sort(['sqft_living', 'price'])
set_2 = set_2.sort(['sqft_living', 'price'])
set_3 = set_3.sort(['sqft_living', 'price'])
set_4 = set_4.sort(['sqft_living', 'price'])

poly_set_1 = plr.polynomial_dataframe(set_1['sqft_living'], 15)
poly_set_2 = plr.polynomial_dataframe(set_2['sqft_living'], 15)
poly_set_3 = plr.polynomial_dataframe(set_3['sqft_living'], 15)
poly_set_4 = plr.polynomial_dataframe(set_4['sqft_living'], 15)
output_set_1 = set_1['price']
output_set_2 = set_2['price']
output_set_3 = set_3['price']
output_set_4 = set_4['price']

model_set_1 = LinearRegression()
model_set_2 = LinearRegression()
model_set_3 = LinearRegression()
model_set_4 = LinearRegression()
model_set_1.fit(poly_set_1, output_set_1)
model_set_2.fit(poly_set_2, output_set_2)
model_set_3.fit(poly_set_3, output_set_3)
model_set_4.fit(poly_set_4, output_set_4)

# Plot order 15 fits
plot_fit(model_set_1, poly_set_1, output_set_1)
plot_fit(model_set_2, poly_set_2, output_set_2)
plot_fit(model_set_3, poly_set_3, output_set_3)
plot_fit(model_set_4, poly_set_4, output_set_4)

# Cross validation to select order
MAX_POLYNOMIAL_DEGREE = 15
RSS = np.zeros((MAX_POLYNOMIAL_DEGREE))
val_output = val_data['price']
test_output = test_data['price']

print 'Cross validation over multiple polynomial fits'
for degree in range(1, MAX_POLYNOMIAL_DEGREE+1):
    train_features = plr.polynomial_dataframe(train_data['sqft_living'], degree)
    val_features = plr.polynomial_dataframe(val_data['sqft_living'], degree)
    model = LinearRegression()
    model.fit(train_features, output)
    RSS[degree-1] = plr.get_residual_sum_of_squares(model, val_features, val_output)
    print 'Fitting degree: ' + str(degree) + ' polynomial' + ' (RSS: ' + str(RSS[degree-1]) + ')'

best_degree = np.argmin(RSS) + 1
print 'Minimum Validation RSS: ' + str (np.min(RSS)) + ' for degree: ' + str(np.argmin(RSS)+1)

model = LinearRegression()
train_features = plr.polynomial_dataframe(train_data['sqft_living'], best_degree)
test_features = plr.polynomial_dataframe(test_data['sqft_living'], best_degree)
model.fit(train_features, output)
RSS_test_best = plr.get_residual_sum_of_squares(model, test_features, test_output)

print 'Test RSS with degree ' + str(best_degree) + ': ' + str(RSS_test_best)
plot_fit(model, train_features, output)
