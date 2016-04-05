# Imports
import sframe
import numpy as np
import decision_tree_binary_boosting as dtb
import matplotlib.pyplot as plt

# Read data
loans = sframe.SFrame('../data/Week05/lending-club-data.gl/')

# Preprocess data
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
loans = loans.remove_column('bad_loans')

# Selected features
features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]

# Output variable in training data
target = 'safe_loans'

# Subsample training data
loans = loans[features + [target]]
safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print "Number of safe loans  : %s" % len(safe_loans_raw)
print "Number of risky loans : %s" % len(risky_loans_raw)
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(percentage, seed=1)
loans_data = risky_loans.append(safe_loans)

# Convert categorical data with one hot encoding
categorical_features = []
for feature_name, feature_type in zip(features, loans_data[features].column_types()):
    if feature_type == str:
        categorical_features.append(feature_name)

for feature_name in categorical_features:
    loans_data_one_hot_encoded = loans_data[feature_name].apply(lambda x: {x: 1})
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature_name)
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)
    loans_data.remove_column(feature_name)
    loans_data.add_columns(loans_data_unpacked)

# Create list of feature names, remove output variable name
one_hot_encoded_features = loans_data.column_names()
one_hot_encoded_features.remove(target)

# Convert from SFrame to numpy arrays
train_data, test_data = loans_data.random_split(0.8, seed=1)
train_label = train_data[target].to_numpy()
train_data = train_data.remove_column(target).to_numpy()
test_label = test_data[target].to_numpy()
test_data = test_data.remove_column(target).to_numpy()


example_data_weights = np.array([1.0]*10 + [0.0]*(len(train_label) - 20) + [1.0]*10)
small_data_decision_tree_subset_20 = dtb.weighted_decision_tree_create(train_data, one_hot_encoded_features, train_label, example_data_weights, max_depth=2)
small_data_decision_tree_subset_20_prediction = np.apply_along_axis(lambda x: dtb.classify(small_data_decision_tree_subset_20, x, one_hot_encoded_features), 1, train_data)
small_data_decision_tree_subset_20_accuracy = dtb.accuracy(small_data_decision_tree_subset_20_prediction, train_label)
small_data_decision_tree_subset_20_error = 1 - small_data_decision_tree_subset_20_accuracy
small_data_decision_tree_subset_20_accuracy_ = dtb.accuracy(small_data_decision_tree_subset_20_prediction[example_data_weights==1], train_label[example_data_weights==1])
small_data_decision_tree_subset_20_error_ = 1 - small_data_decision_tree_subset_20_accuracy_


stump_weights_10, tree_stumps_10 = dtb.adaboost_with_tree_stumps(train_data, one_hot_encoded_features, train_label, num_tree_stumps=10)


stump_weights_30, tree_stumps_30 = dtb.adaboost_with_tree_stumps(train_data, one_hot_encoded_features, train_label, num_tree_stumps=30)


train_error_all = []
for n in xrange(1, 31):
    train_predictions = dtb.classify_adaboost(stump_weights_30[:n], tree_stumps_30[:n], train_data, one_hot_encoded_features)
    train_error = 1.0 - dtb.accuracy(train_predictions, train_label)
    train_error_all.append(train_error)
    print "Iteration %s, training error = %s" % (n, train_error_all[n-1])


plt.rcParams['figure.figsize'] = 7, 5
plt.plot(range(1,31), train_error_all, '-', linewidth=4.0, label='Training error')
plt.title('Performance of Adaboost ensemble')
plt.xlabel('# of iterations')
plt.ylabel('Classification error')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size': 16})


test_error_all = []
for n in xrange(1, 31):
    test_predictions = dtb.classify_adaboost(stump_weights_30[:n], tree_stumps_30[:n], test_data, one_hot_encoded_features)
    test_error = 1.0 - dtb.accuracy(test_predictions, test_label)
    test_error_all.append(test_error)
    print "Iteration %s, training error = %s" % (n, test_error_all[n-1])


plt.rcParams['figure.figsize'] = 7, 5
plt.plot(range(1,31), test_error_all, '-', linewidth=4.0, label='Test error')
plt.title('Performance of Adaboost ensemble')
plt.xlabel('# of iterations')
plt.ylabel('Classification error')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size': 16})
plt.show()
