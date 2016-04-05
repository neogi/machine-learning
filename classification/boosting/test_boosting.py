# Imports
import sframe
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Function to plot errors
def make_figure(dim, title, xlabel, ylabel, legend):
    plt.rcParams['figure.figsize'] = dim
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(loc=legend, prop={'size':15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

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
loans = sframe.SFrame('../data/Week05/lending-club-data.gl/')

# Preprocess data
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
loans = loans.remove_column('bad_loans')

# Selected features
features = ['grade',                     # grade of the loan (categorical)
            'sub_grade_num',             # sub-grade of the loan as a number from 0 to 1
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'payment_inc_ratio',         # ratio of the monthly payment to income
            'delinq_2yrs',               # number of delinquincies
             'delinq_2yrs_zero',         # no delinquincies in last 2 years
            'inq_last_6mths',            # number of creditor inquiries in last 6 months
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'open_acc',                  # number of open credit accounts
            'pub_rec',                   # number of derogatory public records
            'pub_rec_zero',              # no derogatory public records
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
            'int_rate',                  # interest rate of the loan
            'total_rec_int',             # interest received to date
            'annual_inc',                # annual income of borrower
            'funded_amnt',               # amount committed to the loan
            'funded_amnt_inv',           # amount committed by investors for the loan
            'installment',               # monthly payment owed by the borrower
           ]

# Output variable in training data
target = 'safe_loans'

# Count the number of rows with missing data
loans, loans_with_na = loans[[target] + features].dropna_split()
num_rows_with_na = loans_with_na.num_rows()
num_rows = loans.num_rows()
print 'Dropping %s observations; keeping %s ' % (num_rows_with_na, num_rows)

# Subsample training data
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
train_data_ = train_data.remove_column(target)
train_data = train_data_.to_numpy()
validation_label = validation_data[target].to_numpy()
validation_data_ = validation_data.remove_column(target)
validation_data = validation_data_.to_numpy()

# Boosting with 5 trees
model_5 = GradientBoostingClassifier(n_estimators=5, max_depth=6, verbose=1)
model_5.fit(train_data, train_label)

# Check performance on sample validation data
sample_prediction = model_5.predict(sample_validation_data)
sample_prediction_proba = model_5.predict_proba(sample_validation_data)

# Check performance on validation data
model_5_validation_prediction_proba = model_5.predict_proba(validation_data)
model_5_validation_accuracy = accuracy(model_5.predict(validation_data), validation_label)
model_5_validation_confusion_matrix = confusion_matrix(validation_label, model_5.predict(validation_data))
validation_data_['proba'] = model_5_validation_prediction_proba[:, 1]

# Boosted trees with various number of trees
model_10 = GradientBoostingClassifier(n_estimators=10, max_depth=6, verbose=1)
model_10.fit(train_data, train_label)
model_10_train_accuracy = accuracy(model_10.predict(train_data), train_label)
model_10_validation_accuracy = accuracy(model_10.predict(validation_data), validation_label)
model_10_train_error = 1 - model_10_train_accuracy
model_10_validation_error = 1 - model_10_validation_accuracy

model_50 = GradientBoostingClassifier(n_estimators=50, max_depth=6, verbose=1)
model_50.fit(train_data, train_label)
model_50_train_accuracy = accuracy(model_50.predict(train_data), train_label)
model_50_validation_accuracy = accuracy(model_50.predict(validation_data), validation_label)
model_50_train_error = 1 - model_50_train_accuracy
model_50_validation_error = 1 - model_50_validation_accuracy

model_100 = GradientBoostingClassifier(n_estimators=100, max_depth=6, verbose=1)
model_100.fit(train_data, train_label)
model_100_train_accuracy = accuracy(model_100.predict(train_data), train_label)
model_100_validation_accuracy = accuracy(model_100.predict(validation_data), validation_label)
model_100_train_error = 1 - model_100_train_accuracy
model_100_validation_error = 1 - model_100_validation_accuracy

model_200 = GradientBoostingClassifier(n_estimators=200, max_depth=6, verbose=1)
model_200.fit(train_data, train_label)
model_200_train_accuracy = accuracy(model_200.predict(train_data), train_label)
model_200_validation_accuracy = accuracy(model_200.predict(validation_data), validation_label)
model_200_train_error = 1 - model_200_train_accuracy
model_200_validation_error = 1 - model_200_validation_accuracy

model_500 = GradientBoostingClassifier(n_estimators=500, max_depth=6, verbose=1)
model_500.fit(train_data, train_label)
model_500_train_accuracy = accuracy(model_500.predict(train_data), train_label)
model_500_validation_accuracy = accuracy(model_500.predict(validation_data), validation_label)
model_500_train_error = 1 - model_500_train_accuracy
model_500_validation_error = 1 - model_500_validation_accuracy

# Plot training and validation errors vs number of trees
training_errors = [model_10_train_error, model_50_train_error, model_100_train_error, model_200_train_error, model_500_train_error]
validation_errors = [model_10_validation_error, model_50_validation_error, model_100_validation_error, model_200_validation_error, model_500_validation_error]
plt.plot([10, 50, 100, 200, 500], training_errors, linewidth=4.0, label='Training error')
plt.plot([10, 50, 100, 200, 500], validation_errors, linewidth=4.0, label='Validation error')
make_figure(dim=(10,5), title='Error vs number of trees',
            xlabel='Number of trees',
            ylabel='Classification error',
            legend='best')
