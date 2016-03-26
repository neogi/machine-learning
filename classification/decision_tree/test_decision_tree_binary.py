# Imports
import sframe
import numpy as np
import decision_tree_binary as dtree

# Read data
loans = sframe.SFrame('../data/Week04/lending-club-data.gl/')

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
train_data, test_data = loans_data.random_split(.8, seed=1)
train_label = train_data[target].to_numpy()
train_data = train_data.remove_column(target).to_numpy()
test_label = test_data[target].to_numpy()
test_data = test_data.remove_column(target).to_numpy()

# Decision tree with max_depth=6, first assignment
#my_decision_tree = dtree.decision_tree_create(train_data, one_hot_encoded_features, train_label, max_depth=6, min_node_size=0, min_error_reduction=-1.0)
#my_decision_tree_prediction = np.apply_along_axis(lambda x: dtree.classify(my_decision_tree, x, one_hot_encoded_features), 1, test_data)
#my_decision_tree_accuracy = dtree.accuracy(my_decision_tree_prediction, test_label)
#my_decision_tree_error = 1 - my_decision_tree_accuracy

# Decision trees with variety of max_depth, min_node_size and min_error_reduction, second assignment
print "--------"
print "New Tree"
print "--------"
my_decision_tree_new = dtree.decision_tree_create(train_data, one_hot_encoded_features, train_label, max_depth=6, min_node_size=100, min_error_reduction=0.0)
my_decision_tree_new_prediction = np.apply_along_axis(lambda x: dtree.classify(my_decision_tree_new, x, one_hot_encoded_features), 1, test_data)
my_decision_tree_new_accuracy = dtree.accuracy(my_decision_tree_new_prediction, test_label)
my_decision_tree_new_error = 1 - my_decision_tree_new_accuracy

print "--------"
print "Old Tree"
print "--------"
my_decision_tree_old = dtree.decision_tree_create(train_data, one_hot_encoded_features, train_label, max_depth=6, min_node_size=0, min_error_reduction=-1.0)
my_decision_tree_old_prediction = np.apply_along_axis(lambda x: dtree.classify(my_decision_tree_old, x, one_hot_encoded_features), 1, test_data)
my_decision_tree_old_accuracy = dtree.accuracy(my_decision_tree_old_prediction, test_label)
my_decision_tree_old_error = 1 - my_decision_tree_old_accuracy

print "-------"
print "Model 1"
print "-------"
model_1 = dtree.decision_tree_create(train_data, one_hot_encoded_features, train_label, max_depth=2, min_node_size=0, min_error_reduction=-1.0)
model_1_train_prediction = np.apply_along_axis(lambda x: dtree.classify(model_1, x, one_hot_encoded_features), 1, train_data)
model_1_train_accuracy = dtree.accuracy(model_1_train_prediction, train_label)
model_1_train_error = 1 - model_1_train_accuracy
model_1_test_prediction = np.apply_along_axis(lambda x: dtree.classify(model_1, x, one_hot_encoded_features), 1, test_data)
model_1_test_accuracy = dtree.accuracy(model_1_test_prediction, test_label)
model_1_test_error = 1 - model_1_test_accuracy
model_1_leaves = dtree.count_leaves(model_1)

print "-------"
print "Model 2"
print "-------"
model_2 = dtree.decision_tree_create(train_data, one_hot_encoded_features, train_label, max_depth=6, min_node_size=0, min_error_reduction=-1.0)
model_2_train_prediction = np.apply_along_axis(lambda x: dtree.classify(model_2, x, one_hot_encoded_features), 1, train_data)
model_2_train_accuracy = dtree.accuracy(model_2_train_prediction, train_label)
model_2_train_error = 1 - model_2_train_accuracy
model_2_test_prediction = np.apply_along_axis(lambda x: dtree.classify(model_2, x, one_hot_encoded_features), 1, test_data)
model_2_test_accuracy = dtree.accuracy(model_2_test_prediction, test_label)
model_2_test_error = 1 - model_2_test_accuracy
model_2_leaves = dtree.count_leaves(model_2)

print "-------"
print "Model 3"
print "-------"
model_3 = dtree.decision_tree_create(train_data, one_hot_encoded_features, train_label, max_depth=14, min_node_size=0, min_error_reduction=-1.0)
model_3_train_prediction = np.apply_along_axis(lambda x: dtree.classify(model_3, x, one_hot_encoded_features), 1, train_data)
model_3_train_accuracy = dtree.accuracy(model_3_train_prediction, train_label)
model_3_train_error = 1 - model_3_train_accuracy
model_3_test_prediction = np.apply_along_axis(lambda x: dtree.classify(model_3, x, one_hot_encoded_features), 1, test_data)
model_3_test_accuracy = dtree.accuracy(model_3_test_prediction, test_label)
model_3_test_error = 1 - model_3_test_accuracy
model_3_leaves = dtree.count_leaves(model_3)

print "-------"
print "Model 4"
print "-------"
model_4 = dtree.decision_tree_create(train_data, one_hot_encoded_features, train_label, max_depth=6, min_node_size=0, min_error_reduction=-1.0)
model_4_train_prediction = np.apply_along_axis(lambda x: dtree.classify(model_4, x, one_hot_encoded_features), 1, train_data)
model_4_train_accuracy = dtree.accuracy(model_4_train_prediction, train_label)
model_4_train_error = 1 - model_4_train_accuracy
model_4_test_prediction = np.apply_along_axis(lambda x: dtree.classify(model_4, x, one_hot_encoded_features), 1, test_data)
model_4_test_accuracy = dtree.accuracy(model_4_test_prediction, test_label)
model_4_test_error = 1 - model_4_test_accuracy
model_4_leaves = dtree.count_leaves(model_4)

print "-------"
print "Model 5"
print "-------"
model_5 = dtree.decision_tree_create(train_data, one_hot_encoded_features, train_label, max_depth=6, min_node_size=0, min_error_reduction=0.0)
model_5_train_prediction = np.apply_along_axis(lambda x: dtree.classify(model_5, x, one_hot_encoded_features), 1, train_data)
model_5_train_accuracy = dtree.accuracy(model_5_train_prediction, train_label)
model_5_train_error = 1 - model_5_train_accuracy
model_5_test_prediction = np.apply_along_axis(lambda x: dtree.classify(model_5, x, one_hot_encoded_features), 1, test_data)
model_5_test_accuracy = dtree.accuracy(model_5_test_prediction, test_label)
model_5_test_error = 1 - model_5_test_accuracy
model_5_leaves = dtree.count_leaves(model_5)

print "-------"
print "Model 6"
print "-------"
model_6 = dtree.decision_tree_create(train_data, one_hot_encoded_features, train_label, max_depth=6, min_node_size=0, min_error_reduction=5.0)
model_6_train_prediction = np.apply_along_axis(lambda x: dtree.classify(model_6, x, one_hot_encoded_features), 1, train_data)
model_6_train_accuracy = dtree.accuracy(model_6_train_prediction, train_label)
model_6_train_error = 1 - model_6_train_accuracy
model_6_test_prediction = np.apply_along_axis(lambda x: dtree.classify(model_6, x, one_hot_encoded_features), 1, test_data)
model_6_test_accuracy = dtree.accuracy(model_6_test_prediction, test_label)
model_6_test_error = 1 - model_6_test_accuracy
model_6_leaves = dtree.count_leaves(model_6)

print "-------"
print "Model 7"
print "-------"
model_7 = dtree.decision_tree_create(train_data, one_hot_encoded_features, train_label, max_depth=6, min_node_size=0, min_error_reduction=-1)
model_7_train_prediction = np.apply_along_axis(lambda x: dtree.classify(model_7, x, one_hot_encoded_features), 1, train_data)
model_7_train_accuracy = dtree.accuracy(model_7_train_prediction, train_label)
model_7_train_error = 1 - model_7_train_accuracy
model_7_test_prediction = np.apply_along_axis(lambda x: dtree.classify(model_7, x, one_hot_encoded_features), 1, test_data)
model_7_test_accuracy = dtree.accuracy(model_7_test_prediction, test_label)
model_7_test_error = 1 - model_7_test_accuracy
model_7_leaves = dtree.count_leaves(model_7)

print "-------"
print "Model 8"
print "-------"
model_8 = dtree.decision_tree_create(train_data, one_hot_encoded_features, train_label, max_depth=6, min_node_size=2000, min_error_reduction=-1)
model_8_train_prediction = np.apply_along_axis(lambda x: dtree.classify(model_8, x, one_hot_encoded_features), 1, train_data)
model_8_train_accuracy = dtree.accuracy(model_8_train_prediction, train_label)
model_8_train_error = 1 - model_8_train_accuracy
model_8_test_prediction = np.apply_along_axis(lambda x: dtree.classify(model_8, x, one_hot_encoded_features), 1, test_data)
model_8_test_accuracy = dtree.accuracy(model_8_test_prediction, test_label)
model_8_test_error = 1 - model_8_test_accuracy
model_8_leaves = dtree.count_leaves(model_8)

print "-------"
print "Model 9"
print "-------"
model_9 = dtree.decision_tree_create(train_data, one_hot_encoded_features, train_label, max_depth=6, min_node_size=50000, min_error_reduction=-1)
model_9_train_prediction = np.apply_along_axis(lambda x: dtree.classify(model_9, x, one_hot_encoded_features), 1, train_data)
model_9_train_accuracy = dtree.accuracy(model_9_train_prediction, train_label)
model_9_train_error = 1 - model_9_train_accuracy
model_9_test_prediction = np.apply_along_axis(lambda x: dtree.classify(model_9, x, one_hot_encoded_features), 1, test_data)
model_9_test_accuracy = dtree.accuracy(model_9_test_prediction, test_label)
model_9_test_error = 1 - model_9_test_accuracy
model_9_leaves = dtree.count_leaves(model_9)
