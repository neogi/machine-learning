# Imports
import sframe
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Function to remove punctuations
def remove_punctuation(text):
    return string.translate(text, None, string.punctuation)

# Function to compute sigmoid response
def sigmoid(x):
    return 1.0/(1 + np.exp(-x))

# Function to apply thresholds to predicted probabilities
def apply_threshold(probabilities, threshold):
    prediction = np.where(probabilities >= threshold, +1, -1)
    return prediction

# Function to plot precision-recall plot
def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})

# Read data
products = sframe.SFrame('../data/Week06/amazon_baby.gl/')
products['review_clean'] = products['review'].apply(remove_punctuation)

# Discard rating with value 3; these are treated as neither negative nor positive
products = products[products['rating'] != 3]
products['sentiment'] = products['rating'].apply(lambda rating: +1 if rating > 3 else -1)

# Split data into training and testing sets
train_data, test_data = products.random_split(fraction=0.8, seed=1)

# Create training and test matrices from the corresponding data using a CountVectorizer
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(train_data['review_clean'])
test_matrix = vectorizer.transform(test_data['review_clean'])
words = vectorizer.get_feature_names()

# Create a logistic regression model
sentiment_model = LogisticRegression()
sentiment_model.fit(train_matrix, train_data['sentiment'])

# True and predicted classes
y_true = test_data['sentiment'].to_numpy()
y_pred = sentiment_model.predict(test_matrix)

# Accuracy of the sentiment model
sentiment_model_accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)

# Baseline accuracy of the sentiment model
baseline_accuracy = len(test_data[test_data['sentiment'] == 1])/float(len(test_data))

# Confusion matrix on test data using the sentiment model
sentiment_model_confusion_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
print ' target_label | predicted_label | count '
print '--------------+-----------------+-------'
for i, target_label in enumerate(sentiment_model.classes_):
    for j, predicted_label in enumerate(sentiment_model.classes_):
        print '{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, sentiment_model_confusion_matrix[i,j])

# Recall of the sentiment model
sentiment_model_recall = recall_score(y_true=y_true, y_pred=y_pred)

# Precision of the sentiment model
sentiment_model_precision = precision_score(y_true=y_true, y_pred=y_pred)

# Probabilities from the sentiment model
probabilities = sentiment_model.predict_proba(test_matrix)[:,1]

# Number of positive predicitons using thresholds of 0.5 and 0.9
positive_05 = sum(apply_threshold(probabilities, 0.5))
positive_09 = sum(apply_threshold(probabilities, 0.9))

# Recall and precision using thresholds of 0.5 and 0.9
sentiment_model_recall_05 = recall_score(y_true=y_true, y_pred=apply_threshold(probabilities, 0.5))
sentiment_model_precision_05 = precision_score(y_true=y_true, y_pred=apply_threshold(probabilities, 0.5))
sentiment_model_recall_09 = recall_score(y_true=y_true, y_pred=apply_threshold(probabilities, 0.9))
sentiment_model_precision_09 = precision_score(y_true=y_true, y_pred=apply_threshold(probabilities, 0.9))

# Plot precision-recall curve for the sentiment model
threshold_values = np.linspace(0.5, 1, num=100)
recall_all = []
precision_all =[]
for threshold in threshold_values[0:99]:
    sentiment_model_recall_ = recall_score(y_true=y_true, y_pred=apply_threshold(probabilities, threshold))
    sentiment_model_precision_ = precision_score(y_true=y_true, y_pred=apply_threshold(probabilities, threshold))
    recall_all.append(sentiment_model_recall_)
    precision_all.append(sentiment_model_precision_)
plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')

# Confusion matrix for the sentiment model using a threshold of 0.98
sentiment_model_confusion_matrix_098 = confusion_matrix(y_true, apply_threshold(probabilities, 0.98))
print ' target_label | predicted_label | count '
print '--------------+-----------------+-------'
for i, target_label in enumerate(sentiment_model.classes_):
    for j, predicted_label in enumerate(sentiment_model.classes_):
        print '{0:^13} | {1:^15} | {2:5d}'.format(target_label, predicted_label, sentiment_model_confusion_matrix_098[i,j])

# Sentiment model behaviour on baby product reviews
baby_reviews = test_data[test_data['name'].apply(lambda x: 'baby' in x.lower())]
baby_y_true = baby_reviews['sentiment'].to_numpy()
baby_matrix = vectorizer.transform(baby_reviews['review_clean'])
baby_probabilities = sentiment_model.predict_proba(baby_matrix)[:,1]

baby_recall_all = []
baby_precision_all =[]
for threshold in threshold_values[0:99]:
    sentiment_model_recall_ = recall_score(y_true=baby_y_true, y_pred=apply_threshold(baby_probabilities, threshold))
    sentiment_model_precision_ = precision_score(y_true=baby_y_true, y_pred=apply_threshold(baby_probabilities, threshold))
    baby_recall_all.append(sentiment_model_recall_)
    baby_precision_all.append(sentiment_model_precision_)
plot_pr_curve(baby_precision_all, baby_recall_all, 'Precision recall curve (all)')
