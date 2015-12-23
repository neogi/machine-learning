# Imports
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.linear_model import Lasso

# Function to display features with non-zero coeff
def show_features(model):
    print 'Non-zero coeff: ' + str(np.count_nonzero(model.coef_))
    for idx in np.nonzero(model.coef_)[0]:
        print 'Feature ' + str(idx) + ': ' + all_features[idx]

# Function to fit model and evaluate RSS for a set of l1_penalties
def fit_predict_model(l1_penalty):
    RSS = np.zeros((len(l1_penalty)))
    num_nonzero_coeff = np.zeros((len(l1_penalty)))
    idx = 0
    for l1_penalty_choice in l1_penalty:
        model = Lasso(alpha=l1_penalty_choice, normalize=True)
        model.fit(training[all_features], training['price'])
        predicted_price = model.predict(validation[all_features])
        RSS[idx] = np.sum((predicted_price - validation['price'])**2)
        num_nonzero_coeff[idx] = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)
        idx += 1
    return (RSS, num_nonzero_coeff, model)

# Read data
dtype_dict = {'bathrooms':float, 'waterfront':int,
              'sqft_above':int, 'sqft_living15':float,
              'grade':int, 'yr_renovated':int,
              'price':float, 'bedrooms':float,
              'zipcode':str, 'long':float,
              'sqft_lot15':float, 'sqft_living':float,
              'floors':float, 'condition':int,
              'lat':float, 'date':str,
              'sqft_basement':int, 'yr_built':int,
              'id':str, 'sqft_lot':int, 'view':int}
sales = pd.read_csv('../data/Week05/kc_house_data.csv', dtype=dtype_dict)
all_features = ['bedrooms', 'bedrooms_square', 'bathrooms',
                'sqft_living', 'sqft_living_sqrt', 'sqft_lot',
                'sqft_lot_sqrt', 'floors', 'floors_square',
                'waterfront', 'view', 'condition', 'grade',
                'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']

# Add additional features
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors_square'] = sales['floors']*sales['floors']

# Lasso Regression, L1=5e2
model_all = Lasso(alpha=5e2, normalize=True)
model_all.fit(sales[all_features], sales['price'])
show_features(model_all)
print ''

# Read training and test data
testing = pd.read_csv('../data/Week05/wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('../data/Week05/wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('../data/Week05/wk3_kc_house_valid_data.csv', dtype=dtype_dict)

# Add addtional features
testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']

# Select best L1 penalty over a wide range
print 'Selecting best L1 penalty (wide range)'
l1_penalty = np.logspace(1, 7, num=13)
(RSS, num_nonzero_coeff, model) = fit_predict_model(l1_penalty)
    
best_l1_penalty_choice = l1_penalty[np.argmin(RSS)]
print 'Best L1 penalty: ' + str(best_l1_penalty_choice)
print 'Non-zero coeff at best L1 penalty: ' + str(num_nonzero_coeff[np.argmin(RSS)])
print ''

# Select best L1 penalty over a narrow range
print 'Selecting best L1 penalty (narrow range)'
l1_penalty = np.logspace(1, 4, num=20)
(RSS, num_nonzero_coeff, model) = fit_predict_model(l1_penalty)
    
best_l1_penalty_choice = l1_penalty[np.argmin(RSS)]
print 'Best L1 penalty: ' + str(best_l1_penalty_choice)
print 'Non-zero coeff at best L1 penalty: ' + str(num_nonzero_coeff[np.argmin(RSS)])
print ''

# Find L1 penalty bounds such the max non-zero coeff is 7
MAX_NONZERO_COEFF = 7
min_coeff = num_nonzero_coeff[np.less(num_nonzero_coeff, [MAX_NONZERO_COEFF])][0]
max_coeff = num_nonzero_coeff[np.greater(num_nonzero_coeff, [MAX_NONZERO_COEFF])][-1]
upper_bound = l1_penalty[np.equal(num_nonzero_coeff, [min_coeff])][0]
lower_bound = l1_penalty[np.equal(num_nonzero_coeff, [max_coeff])][-1]

print 'Selecting best L1 penalty within lower and upper bounds'
l1_penalty = np.linspace(lower_bound, upper_bound, num=20)
(RSS, num_nonzero_coeff, model) = fit_predict_model(l1_penalty)

reduced_l1_penalty_choice = l1_penalty[np.equal(num_nonzero_coeff, [MAX_NONZERO_COEFF])]
reduced_RSS = RSS[np.equal(num_nonzero_coeff, [MAX_NONZERO_COEFF])]
best_l1_penalty_choice = reduced_l1_penalty_choice[np.argmin(reduced_RSS)]
print 'Best L1 penalty: ' + str(best_l1_penalty_choice)
print 'Non-zero coeff at best L1 penalty: ' + str(num_nonzero_coeff[np.argmin(RSS)])
print ''

model = Lasso(alpha=best_l1_penalty_choice, normalize=True)
model.fit(training[all_features], training['price'])
show_features(model)
print ''
