# Imports
import sframe
import numpy as np
import matplotlib.pyplot as plt
import logistic_regression_mle_L2 as lr_mle_l2

# Read data
products = sframe.SFrame('../data/Week02/amazon_baby_subset.gl/')

# Set of important words; these will be the features
list_of_words = ["baby", "one", "great", "love", "use", "would", "like", "easy", "little", "seat", "old", "well", "get", "also", "really", "son", "time", "bought", "product", "good", "daughter", "much", "loves", "stroller", "put", "months", "car", "still", "back", "used", "recommend", "first", "even", "perfect", "nice", "bag", "two", "using", "got", "fit", "around", "diaper", "enough", "month", "price", "go", "could", "soft", "since", "buy", "room", "works", "made", "child", "keep", "size", "small", "need", "year", "big", "make", "take", "easily", "think", "crib", "clean", "way", "quality", "thing", "better", "without", "set", "new", "every", "cute", "best", "bottles", "work", "purchased", "right", "lot", "side", "happy", "comfortable", "toy", "able", "kids", "bit", "night", "long", "fits", "see", "us", "another", "play", "day", "money", "monitor", "tried", "thought", "never", "item", "hard", "plastic", "however", "disappointed", "reviews", "something", "going", "pump", "bottle", "cup", "waste", "return", "amazon", "different", "top", "want", "problem", "know", "water", "try", "received", "sure", "times", "chair", "find", "hold", "gate", "open", "bottom", "away", "actually", "cheap", "worked", "getting", "ordered", "came", "milk", "bad", "part", "worth", "found", "cover", "many", "design", "looking", "weeks", "say", "wanted", "look", "place", "purchase", "looks", "second", "piece", "box", "pretty", "trying", "difficult", "together", "though", "give", "started", "anything", "last", "company", "come", "returned", "maybe", "took", "broke", "makes", "stay", "instead", "idea", "head", "said", "less", "went", "working", "high", "unit", "seems", "picture", "completely", "wish", "buying", "babies", "won", "tub", "almost", "either"]

# The label
label = ['sentiment']

# Remove punctuations
products['review_clean'] = products['review'].apply(lr_mle_l2.remove_punctuation)

# For each important word add a new column and determine count of that word in all reviews
for word in list_of_words:
    products[word] = products['review_clean'].apply(lambda x: x.split().count(word))

# Check if the word 'perfect' exists in a review
products['contains_perfect'] = products['perfect'].apply(lambda x: +1 if x > 0 else 0)
reviews_with_perfect = sum(products['contains_perfect'])

# Split data in to training and validation sets
train_data, validation_data = products.random_split(.8, seed=2)

# Obtain feature matrix and label array from the products SFrame
(train_feature_matrix, train_label_array) = lr_mle_l2.get_data(train_data, list_of_words, label)
(validation_feature_matrix, validation_label_array) = lr_mle_l2.get_data(validation_data, list_of_words, label)
train_label_array = train_label_array[:, 0]
validation_label_array = validation_label_array[:, 0]

# Set parameters for logistic regression
step_size = 5e-6
max_iter = 501
initial_coefficients = np.zeros(len(list_of_words) + 1)

# L2 penalty = 0
l2_penalty = 0.0
coefficients_0_penalty = lr_mle_l2.logistic_regression(train_feature_matrix, train_label_array, initial_coefficients, step_size, l2_penalty, max_iter)
predictions_0_penalty = lr_mle_l2.predict(train_feature_matrix, coefficients_0_penalty)
accuracy_0_penalty = lr_mle_l2.accuracy(predictions_0_penalty, train_label_array)
predictions_0_penalty_v = lr_mle_l2.predict(validation_feature_matrix, coefficients_0_penalty)
accuracy_0_penalty_v = lr_mle_l2.accuracy(predictions_0_penalty_v, validation_label_array)

# L2 penalty = 4
l2_penalty = 4.0
coefficients_4_penalty = lr_mle_l2.logistic_regression(train_feature_matrix, train_label_array, initial_coefficients, step_size, l2_penalty, max_iter)
predictions_4_penalty = lr_mle_l2.predict(train_feature_matrix, coefficients_4_penalty)
accuracy_4_penalty = lr_mle_l2.accuracy(predictions_4_penalty, train_label_array)
predictions_4_penalty_v = lr_mle_l2.predict(validation_feature_matrix, coefficients_4_penalty)
accuracy_4_penalty_v = lr_mle_l2.accuracy(predictions_4_penalty_v, validation_label_array)

# L2 penalty = 10
l2_penalty = 10.0
coefficients_10_penalty = lr_mle_l2.logistic_regression(train_feature_matrix, train_label_array, initial_coefficients, step_size, l2_penalty, max_iter)
predictions_10_penalty = lr_mle_l2.predict(train_feature_matrix, coefficients_10_penalty)
accuracy_10_penalty = lr_mle_l2.accuracy(predictions_10_penalty, train_label_array)
predictions_10_penalty_v = lr_mle_l2.predict(validation_feature_matrix, coefficients_10_penalty)
accuracy_10_penalty_v = lr_mle_l2.accuracy(predictions_10_penalty_v, validation_label_array)

# L2 penalty = 1e2
l2_penalty = 1.0e2
coefficients_1e2_penalty = lr_mle_l2.logistic_regression(train_feature_matrix, train_label_array, initial_coefficients, step_size, l2_penalty, max_iter)
predictions_1e2_penalty = lr_mle_l2.predict(train_feature_matrix, coefficients_1e2_penalty)
accuracy_1e2_penalty = lr_mle_l2.accuracy(predictions_1e2_penalty, train_label_array)
predictions_1e2_penalty_v = lr_mle_l2.predict(validation_feature_matrix, coefficients_1e2_penalty)
accuracy_1e2_penalty_v = lr_mle_l2.accuracy(predictions_1e2_penalty_v, validation_label_array)

# L2 penalty = 1e3
l2_penalty = 1.0e3
coefficients_1e3_penalty = lr_mle_l2.logistic_regression(train_feature_matrix, train_label_array, initial_coefficients, step_size, l2_penalty, max_iter)
predictions_1e3_penalty = lr_mle_l2.predict(train_feature_matrix, coefficients_1e3_penalty)
accuracy_1e3_penalty = lr_mle_l2.accuracy(predictions_1e3_penalty, train_label_array)
predictions_1e3_penalty_v = lr_mle_l2.predict(validation_feature_matrix, coefficients_1e3_penalty)
accuracy_1e3_penalty_v = lr_mle_l2.accuracy(predictions_1e3_penalty_v, validation_label_array)

# L2 penalty = 1e5
l2_penalty = 1.0e5
coefficients_1e5_penalty = lr_mle_l2.logistic_regression(train_feature_matrix, train_label_array, initial_coefficients, step_size, l2_penalty, max_iter)
predictions_1e5_penalty = lr_mle_l2.predict(train_feature_matrix, coefficients_1e5_penalty)
accuracy_1e5_penalty = lr_mle_l2.accuracy(predictions_1e5_penalty, train_label_array)
predictions_1e5_penalty_v = lr_mle_l2.predict(validation_feature_matrix, coefficients_1e5_penalty)
accuracy_1e5_penalty_v = lr_mle_l2.accuracy(predictions_1e5_penalty_v, validation_label_array)

# Comparing coefficients
coefficients = list(coefficients_0_penalty[1:])
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(list_of_words, coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)

# Plot coefficient paths
plt.rcParams['figure.figsize'] = 10, 6
positive_words = [word for word, coeff in word_coefficient_tuples[0:5]]
negative_words = [word for word, coeff in word_coefficient_tuples[188:193]]
l2_penalty_list = [0, 4, 10, 1e2, 1e3, 1e5]
table = sframe.SFrame()
table['words']=list_of_words
table['0'] = coefficients_0_penalty[1:]
table['4'] = coefficients_4_penalty[1:]
table['10'] = coefficients_10_penalty[1:]
table['1e2'] = coefficients_1e2_penalty[1:]
table['1e3'] = coefficients_1e3_penalty[1:]
table['1e5'] = coefficients_1e5_penalty[1:]
lr_mle_l2.make_coefficient_plot(table, positive_words, negative_words, [0, 4, 10, 1e2, 1e3, 1e5])
