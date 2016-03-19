# Imports
import sframe
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
 
# Function to remove punctuations
def remove_punctuation(text):
    return string.translate(text, None, string.punctuation)
 
# Function to compute sigmoid response
def sigmoid(x):
    return 1.0/(1 + np.exp(-x))
 
# Read data
products = sframe.SFrame('../data/Week01/amazon_baby.gl/')
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
 
# Create a SFrame with words and their corresponding coefficient
sentiment_model_coef_table = sframe.SFrame({'word':words,
                                            'coefficient':sentiment_model.coef_.flatten()})
 
# Sanity check using some sample data
sample_test_data = test_data[10:13]
sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
sample_test_scores = sentiment_model.decision_function(sample_test_matrix)
sample_test_probabilities = sigmoid(sample_test_scores)
 
# Apply the logistic regression model on the test matrix
# Compute scores, compute probabilities, compute predicted sentiment
test_scores = sentiment_model.decision_function(test_matrix)
test_probabilities = sigmoid(test_scores)
test_data['probability'] = test_probabilities
test_data['predicted_score'] = test_scores
test_data['predicted_sentiment'] = test_data['predicted_score'].apply(lambda score: +1 if score > 0.0 else -1)
 
# Sort the test data on the predicted probability
# Get the likely products for the most positive and most negative reviews
test_data.sort('probability', ascending=False)['name'][0:20]
test_data.sort('probability', ascending=True)['name'][0:20]
 
# Compute accuracy for the logistic regression model on the test data
test_data_correct = sum(test_data['predicted_sentiment'] == test_data['sentiment'])
test_data_total = len(test_data)*1.0
test_data_accuracy = test_data_correct/test_data_total
 
# Create training and test matrices from the corresponding data using a CountVectorizer
# Note that only some words are used that might indicate a positive or negative view
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves',
                     'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed',
                     'work', 'product', 'money', 'would', 'return']
vectorizer_subset = CountVectorizer(vocabulary=significant_words)
train_matrix_subset = vectorizer_subset.fit_transform(train_data['review_clean'])
test_matrix_subset = vectorizer_subset.transform(test_data['review_clean'])
 
# Create a logistic regression using the subset of words
sentiment_model_subset = LogisticRegression()
sentiment_model_subset.fit(train_matrix_subset, train_data['sentiment'])
sentiment_model_subset_coef_table = sframe.SFrame({'word':significant_words,
                                                   'coefficient':sentiment_model_subset.coef_.flatten()})
 
# Print out word sentiments in the original and the later logistic regression model
# Compare if the coefficients are positive/negative in both the models for the same word
for w in sentiment_model_subset_coef_table['word']:
    print sentiment_model_coef_table[sentiment_model_coef_table['word'] == w]
    print sentiment_model_subset_coef_table[sentiment_model_subset_coef_table['word'] == w]
 
# Apply the logistic regression model with the subset of words on the test matrix
# Compute scores, compute probabilities, compute predicted sentiment
test_scores_subset = sentiment_model_subset.decision_function(test_matrix_subset)
test_probabilities_subset = sigmoid(test_scores_subset)
test_data['probability_subset'] = test_probabilities_subset
test_data['predicted_score_subset'] = test_scores_subset
test_data['predicted_sentiment_subset'] = test_data['predicted_score_subset'].apply(lambda score: +1 if score > 0.0 else -1)
 
# Compute accuracy for the logistic regression model with the subset of words on the test data
test_data_subset_correct = sum(test_data['predicted_sentiment_subset'] == test_data['sentiment'])
test_data_total = len(test_data)*1.0
test_data_subset_accuracy = test_data_subset_correct/test_data_total
 
# Apply the logistic regression model on the train matrix
# Compute scores, compute probabilities, compute predicted sentiment
train_scores = sentiment_model.decision_function(train_matrix)
train_probabilities = sigmoid(train_scores)
train_data['probability'] = train_probabilities
train_data['predicted_score'] = train_scores
train_data['predicted_sentiment'] = train_data['predicted_score'].apply(lambda score: +1 if score > 0.0 else -1)
 
# Compute accuracy for the logistic regression model on the train data
train_data_correct = sum(train_data['predicted_sentiment'] == train_data['sentiment'])
train_data_total = len(train_data)*1.0
train_data_accuracy = train_data_correct/train_data_total
 
# Apply the logistic regression model with the subset of words on the train matrix
# Compute scores, compute probabilities, compute predicted sentiment
train_scores_subset = sentiment_model_subset.decision_function(train_matrix_subset)
train_probabilities_subset = sigmoid(train_scores_subset)
train_data['probability_subset'] = train_probabilities_subset
train_data['predicted_score_subset'] = train_scores_subset
train_data['predicted_sentiment_subset'] = train_data['predicted_score_subset'].apply(lambda score: +1 if score > 0.0 else -1)
 
# Compute accuracy for the logistic regression model with the subset of words on the train data
train_data_subset_correct = sum(train_data['predicted_sentiment_subset'] == train_data['sentiment'])
train_data_total = len(train_data)*1.0
train_data_subset_accuracy = train_data_subset_correct/train_data_total
 
# The majority of the reviews are positive
# Compare the accuracy of the majority classifier and the logistic regression classfier
train_data_majority_correct = sum(train_data['sentiment'] == 1)
train_data_total = len(train_data)*1.0
train_data_majority_accuracy = train_data_majority_correct/train_data_total
