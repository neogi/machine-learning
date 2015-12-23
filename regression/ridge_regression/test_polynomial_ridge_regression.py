# Imports
import numpy as np
import pandas as pd
import polynomial_ridge_regression as plr
from sklearn.linear_model import LinearRegression, Ridge
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

# Options for Ridge Regression
n = True # Normalize predictors
s = 'auto' # Choose solver automatically
i = True # Fit intercept
t = 0.01 # Tolerance

# Read training and test data
train_data = pd.read_csv('../data/Week04/kc_house_data.csv', dtype=dtype_dict)
test_data = pd.read_csv('../data/Week04/wk3_kc_house_test_data.csv', dtype=dtype_dict)
train_data = train_data.sort(['sqft_living', 'price'])
output = train_data['price']

# Create order 15 polynomial feature, train and predict
poly15_data = plr.polynomial_dataframe(train_data['sqft_living'], 15)

# LS Linear Regression
model1 = LinearRegression(normalize=n, fit_intercept=i)
model1.fit(poly15_data, output)

# Plot model 1 order 15 fit
plot_fit(model1, poly15_data, output)

# Ridge Regression
model2 = Ridge(normalize=n, alpha=1e-5, solver=s, fit_intercept=i, tol=t)
model2.fit(poly15_data, output)

# Plot model 2 order 15 fit
plot_fit(model2, poly15_data, output)

# Compare coefficients of Linear and Ridge Regression
print 'Linear regression coefficients:'
print model1.coef_
print 'Ridge regression coefficients:'
print model2.coef_

# Read other datasets
set_1 = pd.read_csv('../data/Week04/wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
set_2 = pd.read_csv('../data/Week04/wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
set_3 = pd.read_csv('../data/Week04/wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
set_4 = pd.read_csv('../data/Week04/wk3_kc_house_set_4_data.csv', dtype=dtype_dict)
set_1 = set_1.sort(['sqft_living', 'price'])
set_2 = set_2.sort(['sqft_living', 'price'])
set_3 = set_3.sort(['sqft_living', 'price'])
set_4 = set_4.sort(['sqft_living', 'price'])

poly15_set_1 = plr.polynomial_dataframe(set_1['sqft_living'], 15)
poly15_set_2 = plr.polynomial_dataframe(set_2['sqft_living'], 15)
poly15_set_3 = plr.polynomial_dataframe(set_3['sqft_living'], 15)
poly15_set_4 = plr.polynomial_dataframe(set_4['sqft_living'], 15)
output_set_1 = set_1['price']
output_set_2 = set_2['price']
output_set_3 = set_3['price']
output_set_4 = set_4['price']

# Ridge Regression L2=1e-5
l2_penalty = 1e-5
model1_set_1 = Ridge(normalize=n, alpha=l2_penalty, solver=s, fit_intercept=i, tol=t)
model1_set_2 = Ridge(normalize=n, alpha=l2_penalty, solver=s, fit_intercept=i, tol=t)
model1_set_3 = Ridge(normalize=n, alpha=l2_penalty, solver=s, fit_intercept=i, tol=t)
model1_set_4 = Ridge(normalize=n, alpha=l2_penalty, solver=s, fit_intercept=i, tol=t)
model1_set_1.fit(poly15_set_1, output_set_1)
model1_set_2.fit(poly15_set_2, output_set_2)
model1_set_3.fit(poly15_set_3, output_set_3)
model1_set_4.fit(poly15_set_4, output_set_4)

# Plot model 1 order 15 fits
plot_fit(model1_set_1, poly15_set_1, output_set_1)
plot_fit(model1_set_2, poly15_set_2, output_set_2)
plot_fit(model1_set_3, poly15_set_3, output_set_3)
plot_fit(model1_set_4, poly15_set_4, output_set_4)

# Ridge Regression L2=1e5
l2_penalty = 1e5
model2_set_1 = Ridge(normalize=n, alpha=l2_penalty, solver=s, fit_intercept=i, tol=t)
model2_set_2 = Ridge(normalize=n, alpha=l2_penalty, solver=s, fit_intercept=i, tol=t)
model2_set_3 = Ridge(normalize=n, alpha=l2_penalty, solver=s, fit_intercept=i, tol=t)
model2_set_4 = Ridge(normalize=n, alpha=l2_penalty, solver=s, fit_intercept=i, tol=t)
model2_set_1.fit(poly15_set_1, output_set_1)
model2_set_2.fit(poly15_set_2, output_set_2)
model2_set_3.fit(poly15_set_3, output_set_3)
model2_set_4.fit(poly15_set_4, output_set_4)

# Plot model 2 order 15 fits
plot_fit(model2_set_1, poly15_set_1, output_set_1)
plot_fit(model2_set_2, poly15_set_2, output_set_2)
plot_fit(model2_set_3, poly15_set_3, output_set_3)
plot_fit(model2_set_4, poly15_set_4, output_set_4)

# Compare coefficients for different L2 values
print 'Model 1 (power_1)'
print model1_set_1.coef_[0], model1_set_2.coef_[0], model1_set_3.coef_[0], model1_set_4.coef_[0]
print 'Model 2 (power_1)'
print model2_set_1.coef_[0], model2_set_2.coef_[0], model2_set_3.coef_[0], model1_set_4.coef_[0]

# Cross validation to select l2 penalty
print 'Selecting best L2 penalty using 10 fold cross validation'
MAX_POLYNOMIAL_DEGREE = 15
train_valid_shuffled_data = pd.read_csv('../data/Week04/wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
poly15_train_valid_shuffled_data = plr.polynomial_dataframe(train_valid_shuffled_data['sqft_living'], MAX_POLYNOMIAL_DEGREE)
output = train_valid_shuffled_data['price']

l2_penalty = np.logspace(1, 7, num=13)
k = 10

average_error = np.zeros((len(l2_penalty)))
idx = 0
for l2_penalty_choice in l2_penalty:
    print 'Evaluating L2: ' + str(l2_penalty_choice)
    model = Ridge(normalize=n, alpha=l2_penalty_choice, solver=s, fit_intercept=i, tol=t)
    average_error[idx] = plr.k_fold_cross_validation(k,
                                                     l2_penalty_choice,
                                                     poly15_train_valid_shuffled_data,
                                                     output,
                                                     model)
    print 'Validation error: ' + str(average_error[idx])
    idx += 1
    
best_l2_penalty_choice = l2_penalty[np.argmin(average_error)]
print 'Best L2 penalty: ' + str(best_l2_penalty_choice)

model = Ridge(normalize=n, alpha=best_l2_penalty_choice, solver=s, fit_intercept=i, tol=t)
model.fit(poly15_train_valid_shuffled_data, output)

test_error = plr.get_residual_sum_of_squares(model,
                                             plr.polynomial_dataframe(test_data['sqft_living'], MAX_POLYNOMIAL_DEGREE),
                                             test_data['price'])

print 'Test error: ' + str(test_error) + ' at L2: ' + str(best_l2_penalty_choice)
