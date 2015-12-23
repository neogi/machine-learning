# Imports
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Read training and test data
train_data = pd.read_csv('../data/Week02/kc_house_train_data.csv')
test_data = pd.read_csv('../data/Week02/kc_house_test_data.csv')

# Add new features to train data
train_data['bedrooms_squared'] = train_data['bedrooms'] ** 2
train_data['bed_bath_rooms'] = train_data['bedrooms'] * train_data['bathrooms']
train_data['log_sqft_living'] = np.log(train_data['sqft_living'])
train_data['lat_plus_long'] = train_data['lat'] + train_data['long']

# Add new features to test data
test_data['bedrooms_squared'] = test_data['bedrooms'] ** 2
test_data['bed_bath_rooms'] = test_data['bedrooms'] * test_data['bathrooms']
test_data['log_sqft_living'] = np.log(test_data['sqft_living'])
test_data['lat_plus_long'] = test_data['lat'] + test_data['long']

# Extract different feature sets and target
features1 = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
features2 = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms']
features3 = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms', 'log_sqft_living', 'lat_plus_long']

# Feature sets from train data
y = train_data['price']
x1 = train_data[features1]
x2 = train_data[features2]
x3 = train_data[features3]

# Train models on different feature sets
model1 = LinearRegression()
model2 = LinearRegression()
model3 = LinearRegression()
model1.fit(x1, y)
model2.fit(x2, y)
model3.fit(x3, y)

# Predict using train data and compute RSS
y1 = model1.predict(x1)
y2 = model2.predict(x2)
y3 = model3.predict(x3)
RSS1 = np.sum((y - y1) ** 2)
RSS2 = np.sum((y - y2) ** 2)
RSS3 = np.sum((y - y3) ** 2)

# Feature sets from test data
yt = test_data['price']
t1 = test_data[features1]
t2 = test_data[features2]
t3 = test_data[features3]

# Predict using train data and compute RSS
yt1 = model1.predict(t1)
yt2 = model2.predict(t2)
yt3 = model3.predict(t3)
RSSt1 = np.sum((yt - yt1) ** 2)
RSSt2 = np.sum((yt - yt2) ** 2)
RSSt3 = np.sum((yt - yt3) ** 2)
