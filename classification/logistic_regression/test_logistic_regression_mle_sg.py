# Imports
import sframe
import numpy as np
import logistic_regression_mle_sg as lr_mle_sg
import matplotlib.pyplot as plt

# Function to plot likelihood curves
def make_plot(log_likelihood_all, len_data, batch_size, smoothing_window=1, label=''):
    plt.rcParams.update({'figure.figsize': (9,5)})
    log_likelihood_all_ma = np.convolve(np.array(log_likelihood_all), \
                                        np.ones((smoothing_window,))/smoothing_window, mode='valid')
    plt.plot(np.array(range(smoothing_window-1, len(log_likelihood_all)))*float(batch_size)/len_data,
             log_likelihood_all_ma, linewidth=4.0, label=label)
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.xlabel('# of passes over data')
    plt.ylabel('Average log likelihood per data point')
    plt.legend(loc='lower right', prop={'size':14})

# Read data
products = sframe.SFrame('../data/Week02/amazon_baby_subset.gl/')

# Set of important words; these will be the features
list_of_words = ["baby", "one", "great", "love", "use", "would", "like", "easy", "little", "seat", "old", "well", "get", "also", "really", "son", "time", "bought", "product", "good", "daughter", "much", "loves", "stroller", "put", "months", "car", "still", "back", "used", "recommend", "first", "even", "perfect", "nice", "bag", "two", "using", "got", "fit", "around", "diaper", "enough", "month", "price", "go", "could", "soft", "since", "buy", "room", "works", "made", "child", "keep", "size", "small", "need", "year", "big", "make", "take", "easily", "think", "crib", "clean", "way", "quality", "thing", "better", "without", "set", "new", "every", "cute", "best", "bottles", "work", "purchased", "right", "lot", "side", "happy", "comfortable", "toy", "able", "kids", "bit", "night", "long", "fits", "see", "us", "another", "play", "day", "money", "monitor", "tried", "thought", "never", "item", "hard", "plastic", "however", "disappointed", "reviews", "something", "going", "pump", "bottle", "cup", "waste", "return", "amazon", "different", "top", "want", "problem", "know", "water", "try", "received", "sure", "times", "chair", "find", "hold", "gate", "open", "bottom", "away", "actually", "cheap", "worked", "getting", "ordered", "came", "milk", "bad", "part", "worth", "found", "cover", "many", "design", "looking", "weeks", "say", "wanted", "look", "place", "purchase", "looks", "second", "piece", "box", "pretty", "trying", "difficult", "together", "though", "give", "started", "anything", "last", "company", "come", "returned", "maybe", "took", "broke", "makes", "stay", "instead", "idea", "head", "said", "less", "went", "working", "high", "unit", "seems", "picture", "completely", "wish", "buying", "babies", "won", "tub", "almost", "either"]

# The label
label = ['sentiment']

# Remove punctuations
products['review_clean'] = products['review'].apply(lr_mle_sg.remove_punctuation)

# For each important word add a new column and determine count of that word in all reviews
for word in list_of_words:
    products[word] = products['review_clean'].apply(lambda x: x.split().count(word))

train_data, validation_data = products.random_split(.9, seed=1)

# Obtain train and validation matrices and label arrays from the corresponding SFrames
(train_matrix, train_label) = lr_mle_sg.get_data(train_data, list_of_words, label)
train_label = train_label[:, 0]
(validation_matrix, validation_label) = lr_mle_sg.get_data(validation_data, list_of_words, label)
validation_label = validation_label[:, 0]

# Evaluate logistic regression using stochastic gradient ascent
initial_coefficients = np.zeros(194)
step_size = 5.0e-1
batch_size = 1
max_iter = 10
(coefficients, log_likelihood_all) = lr_mle_sg.logistic_regression_SG(train_matrix, train_label, initial_coefficients, step_size, batch_size, max_iter)

initial_coefficients = np.zeros(194)
step_size = 5.0e-1
batch_size = len(train_matrix)
max_iter = 200
(coefficients, log_likelihood_all) = lr_mle_sg.logistic_regression_SG(train_matrix, train_label, initial_coefficients, step_size, batch_size, max_iter)

initial_coefficients = np.zeros(194)
step_size = 1.0e-1
batch_size = 100
num_passes = 10
num_iter = num_passes * int(len(train_matrix)/batch_size)
(coefficients_sgd, log_likelihood_sgd) = lr_mle_sg.logistic_regression_SG(train_matrix, train_label, initial_coefficients, step_size, batch_size, num_iter)

make_plot(log_likelihood_sgd, len_data=len(train_matrix), batch_size=100, label='stochastic gradient, step_size=1e-1')
make_plot(log_likelihood_sgd, len_data=len(train_matrix), batch_size=100, smoothing_window=30, label='stochastic gradient, step_size=1e-1')

# Batch size less than number of training samples
initial_coefficients = np.zeros(194)
step_size = 1.0e-1
batch_size = 100
num_passes = 200
num_iter = num_passes * int(len(train_matrix)/batch_size)
(coefficients_sgd, log_likelihood_sgd) = lr_mle_sg.logistic_regression_SG(train_matrix, train_label, initial_coefficients, step_size, batch_size, num_iter)
make_plot(log_likelihood_sgd, len_data=len(train_matrix), batch_size=100, smoothing_window=30, label='stochastic, step_size=1e-1')

# Batch size equal to number of training samples
initial_coefficients = np.zeros(194)
step_size = 5.0e-1
batch_size = len(train_matrix)
num_passes = 200
num_iter = num_passes * int(len(train_matrix)/batch_size)
(coefficients_batch, log_likelihood_batch) = lr_mle_sg.logistic_regression_SG(train_matrix, train_label, initial_coefficients, step_size, batch_size, num_iter)
make_plot(log_likelihood_batch, len_data=len(train_matrix), batch_size=len(train_matrix), smoothing_window=1, label='batch, step_size=5e-1')

# Evaluate effect of step size on stochastic gradient ascent
initial_coefficients = np.zeros(194)
batch_size = 100
num_passes = 10
num_iter = num_passes * int(len(train_matrix)/batch_size)

coefficients_sgd = {}
log_likelihood_sgd = {}

for step_size in np.logspace(-4, 2, num=7):
    (coefficients_sgd[step_size], log_likelihood_sgd[step_size]) = lr_mle_sg.logistic_regression_SG(train_matrix, train_label, initial_coefficients, step_size, batch_size, num_iter)

for step_size in np.logspace(-4, 2, num=7)[0:6]:
    make_plot(log_likelihood_sgd[step_size], len_data=len(train_data), batch_size=100, smoothing_window=30, label='step_size=%.1e'%step_size)
