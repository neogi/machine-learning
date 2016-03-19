# Imports
import sframe
import numpy as np
import collections
import logistic_regression_mle as lr_mle

# Read data
products = sframe.SFrame('../data/Week02/amazon_baby_subset.gl/')

# Set of important words; these will be the features
list_of_words = ["baby", "one", "great", "love", "use", "would", "like", "easy", "little", "seat", "old", "well", "get", "also", "really", "son", "time", "bought", "product", "good", "daughter", "much", "loves", "stroller", "put", "months", "car", "still", "back", "used", "recommend", "first", "even", "perfect", "nice", "bag", "two", "using", "got", "fit", "around", "diaper", "enough", "month", "price", "go", "could", "soft", "since", "buy", "room", "works", "made", "child", "keep", "size", "small", "need", "year", "big", "make", "take", "easily", "think", "crib", "clean", "way", "quality", "thing", "better", "without", "set", "new", "every", "cute", "best", "bottles", "work", "purchased", "right", "lot", "side", "happy", "comfortable", "toy", "able", "kids", "bit", "night", "long", "fits", "see", "us", "another", "play", "day", "money", "monitor", "tried", "thought", "never", "item", "hard", "plastic", "however", "disappointed", "reviews", "something", "going", "pump", "bottle", "cup", "waste", "return", "amazon", "different", "top", "want", "problem", "know", "water", "try", "received", "sure", "times", "chair", "find", "hold", "gate", "open", "bottom", "away", "actually", "cheap", "worked", "getting", "ordered", "came", "milk", "bad", "part", "worth", "found", "cover", "many", "design", "looking", "weeks", "say", "wanted", "look", "place", "purchase", "looks", "second", "piece", "box", "pretty", "trying", "difficult", "together", "though", "give", "started", "anything", "last", "company", "come", "returned", "maybe", "took", "broke", "makes", "stay", "instead", "idea", "head", "said", "less", "went", "working", "high", "unit", "seems", "picture", "completely", "wish", "buying", "babies", "won", "tub", "almost", "either"]

# The label
label = ['sentiment']

# Remove punctuations
products['review_clean'] = products['review'].apply(lr_mle.remove_punctuation)

# For each important word add a new column and determine count of that word in all reviews
for word in list_of_words:
    products[word] = products['review_clean'].apply(lambda x: x.split().count(word))

# Check if the word 'perfect' exists in a review
products['contains_perfect'] = products['perfect'].apply(lambda x: +1 if x > 0 else 0)
reviews_with_perfect = sum(products['contains_perfect'])

# Obtain feature matrix and label array from the products SFrame
(feature_matrix, label_array) = lr_mle.get_data(products, list_of_words, label)
label_array = label_array[:, 0]

# Set parameters for logistic regression
step_size = 1e-7
max_iter = 301
initial_coefficients = np.zeros(len(list_of_words) + 1)
coefficients = lr_mle.logistic_regression(feature_matrix, label_array, initial_coefficients, step_size, max_iter)
predictions = lr_mle.predict(feature_matrix, coefficients)

# Compute prediction statistics
prediction_counts = collections.Counter(predictions)
prediction_correct = sum((label_array == predictions)*1.0)
prediction_total = len(predictions)
accuracy = prediction_correct/prediction_total

# Determine words and their coefficients
coefficients = list(coefficients[1:])
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(list_of_words, coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)
