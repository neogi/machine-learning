# Imports
import sframe
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
import subprocess

# Function to compute accuracy
def accuracy(prediction, actual):
    """
    Purpose: Compute accuracy
    Input  : Predicted output values, true output values
    Output : Accuracy
    """
    prediction_correct = sum((actual == prediction)*1.0)
    prediction_total = len(prediction)
    accuracy = prediction_correct/prediction_total
    return accuracy

# Read data
loans = sframe.SFrame('../data/Week03/lending-club-data.gl/')

# Preprocess data
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
loans = loans.remove_column('bad_loans')

# Selected features
features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

# Output variable in training data
target = 'safe_loans'                    # prediction target (y) (+1 means safe, -1 is risky)

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
categorical_variables = []
for feat_name, feat_type in zip(loans_data.column_names(), loans_data.column_types()):
    if feat_type == str:
        categorical_variables.append(feat_name)

for feature in categorical_variables:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)
    loans_data.remove_column(feature)
    loans_data.add_columns(loans_data_unpacked)

# Create training and validation data
train_data, validation_data = loans_data.random_split(.8, seed=1)

# Create a small validation sample
validation_safe_loans = validation_data[validation_data[target] == 1]
validation_risky_loans = validation_data[validation_data[target] == -1]
sample_validation_data_risky = validation_risky_loans[0:2]
sample_validation_data_safe = validation_safe_loans[0:2]
sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky)
sample_validation_data_label = sample_validation_data[target].to_numpy()
sample_validation_data = sample_validation_data.remove_column(target).to_numpy()

# Convert from SFrame to numpy arrays
train_label = train_data[target].to_numpy()
train_data = train_data.remove_column(target).to_numpy()
validation_label = validation_data[target].to_numpy()
validation_data = validation_data.remove_column(target).to_numpy()

# Decision tree with max_depth=6
decision_tree_model = DecisionTreeClassifier(max_depth=6)
decision_tree_model.fit(train_data, train_label)
#export_graphviz(decision_tree_model, out_file='decision_tree_model.dot', feature_names=features)
#subprocess.call(['dot', '-Tsvg', 'decision_tree_model.dot', '-o' 'decision_tree_model.svg'])
decision_tree_model_train_accuracy = accuracy(decision_tree_model.predict(train_data), train_label)
decision_tree_model_validation_accuracy = accuracy(decision_tree_model.predict(validation_data), validation_label)
decision_tree_model_validation_confusion_matrix = confusion_matrix(validation_label, decision_tree_model.predict(validation_data), labels=[-1, +1])

# Decision tree with max_depth=2
small_model = DecisionTreeClassifier(max_depth=2)
small_model.fit(train_data, train_label)
#export_graphviz(small_model, out_file='small_model.dot', feature_names=features)
#subprocess.call(['dot', '-Tsvg', 'small_model.dot', '-o' 'small_model.svg'])
small_model_train_accuracy = accuracy(small_model.predict(train_data), train_label)
small_model_validation_accuracy = accuracy(small_model.predict(validation_data), validation_label)

# Decision tree with max_depth=10
big_model = DecisionTreeClassifier(max_depth=10)
big_model.fit(train_data, train_label)
#export_graphviz(big_model, out_file='big_model.dot', feature_names=features)
#subprocess.call(['dot', '-Tsvg', 'big_model.dot', '-o' 'big_model.svg'])
big_model_train_accuracy = accuracy(big_model.predict(train_data), train_label)
big_model_validation_accuracy = accuracy(big_model.predict(validation_data), validation_label)

# Checking output of decision trees using small validation sample
sample_prediction = decision_tree_model.predict(sample_validation_data)
sample_prediction_proba = decision_tree_model.predict_proba(sample_validation_data)
sample_prediction_small = small_model.predict(sample_validation_data)
sample_prediction_proba_small = small_model.predict_proba(sample_validation_data)
