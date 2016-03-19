# Imports
import string
import numpy as np
import matplotlib.pyplot as plt

# Function to remove punctuations
def remove_punctuation(text):
    """
    Purpose: Remove all punctuations from a line of text
    Input  : A line of text
    Output : Text without punctuations
    """
    return string.translate(text, None, string.punctuation)

# Function to compute score
def compute_score(feature_matrix, coefficients):
    """
    Purpose: Compute the dot product of features and their coefficients
    Input  : Feature matrix and a coefficient vector
    Output : Dot product of feature matrix and coefficient vector
    """
    return feature_matrix.dot(coefficients)

# Function to compute probability
def compute_probability(score):
    """
    Purpose: Compute sigmoid response (probabilities) from scores
    Input  : A vector of scores
    Output : A vector of probabilities
    """
    return 1.0/(1 + np.exp(-score))

# Function to compute feature derivative
def compute_feature_derivative(errors, feature, coefficient, l2_penalty, feature_is_constant):
    """
    Purpose: Compute derivative of a feature wrt coefficient
    Input  : Error between true output and predicted output values, feature,
             coefficient, L2 penalty strength, if the feature is constant or not
    Output : Derivative of the feature wrt coefficient
    """
    if not feature_is_constant:
       derivative = feature.T.dot(errors)
    else:
       derivative = feature.T.dot(errors) - 2.0 * l2_penalty * coefficient
    return derivative

# Function to compute log likelihood
def compute_log_likelihood(feature_matrix, sentiment, coefficients, l2_penalty):
    """
    Purpose: Compute Log-Likelihood
    Input  : Feature matrix, coefficients, true output values, L2 penalty strength
    Output : Log-Likelihood
    """
    indicator = (sentiment == +1)
    scores = compute_score(feature_matrix, coefficients)
    log_likelihood = np.sum((indicator - 1) * scores - np.log(1.0 + np.exp(-scores))) - l2_penalty * np.sum(coefficients[1:]**2)
    return log_likelihood

# Function to prepare array from SFrame
def get_data(data_frame, features, label):
    """
    Purpose: Extract features and prepare a feature matrix
             Set the first feature x0 = 1
    Input  : Original Dataframe, list of feature variables, output variable
    Output : Feature matrix, label array
    """
    data_frame['constant'] = 1.0
    features = ['constant'] + features
    features_matrix = data_frame[features].to_numpy()
    if label != None:
        label_array = data_frame[label].to_numpy()
    else:
        label_array = []
    return(features_matrix, label_array)

# Funtion to perform logistic regression
def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, l2_penalty, max_iter):
    """
    Purpose: Perform logistic regression
    Input  : Feature matrix, true output values, initial estimate of coefficients, step size
             L2 penalty strength, maximum number of iterations
    Output : Estimated coefficient vector
    """
    coefficients = np.array(initial_coefficients)
     
    for itr in xrange(max_iter):
        predictions = compute_probability(compute_score(feature_matrix, coefficients))
        indicator = (sentiment == +1)*1.0
        errors = indicator - predictions
         
        for j in xrange(len(coefficients)):
            derivative = compute_feature_derivative(errors, feature_matrix[:, j], coefficients[j], l2_penalty, j)
            coefficients[j] = coefficients[j] + step_size * derivative
     
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients, l2_penalty)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
            (int(np.ceil(np.log10(max_iter))), itr, lp)
     
    return coefficients

# Function to predict classes
def predict(feature_matrix, coefficients):
    """
    Purpose: Predict output values from feature matrix and estimated coefficients
    Input  : Feature matrix, coefficient vector
    Output : Predicted output vector
    """
    scores = compute_score(feature_matrix, coefficients)
    classes = (scores > 0.0)*2
    classes = classes - 1
    return classes

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

# Function to plot coefficient paths
def make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list):
    """
    Purpose: To plot coefficient path vs L2 penalty strength
    Input  : Table of words and their coefficient values for each L2 penalty value,
             list of positive words, list of negative words, list of L2 penalty values
    Output : Plot of coefficient paths for the list of positive and negative words
    """
    cmap_positive = plt.get_cmap('Reds')
    cmap_negative = plt.get_cmap('Blues')
    
    xx = l2_penalty_list
    plt.plot(xx, [0.]*len(xx), '--', lw=1, color='k')
    
    table_positive_words = table.filter_by(positive_words, 'words')
    table_negative_words = table.filter_by(negative_words, 'words')
    del table_positive_words['words']
    del table_negative_words['words']
    
    for i in xrange(len(positive_words)):
        color = cmap_positive(0.8*((i+1)/(len(positive_words)*1.2)+0.15))
        plt.plot(xx, table_positive_words[i:i+1].to_numpy()[0,:],
                 '-', label=positive_words[i], linewidth=4.0, color=color)
        
    for i in xrange(len(negative_words)):
        color = cmap_negative(0.8*((i+1)/(len(negative_words)*1.2)+0.15))
        plt.plot(xx, table_negative_words[i:i+1].to_numpy()[0,:],
                 '-', label=negative_words[i], linewidth=4.0, color=color)
        
    plt.legend(loc='best', ncol=3, prop={'size':16}, columnspacing=0.5)
    plt.axis([1, 1e5, -1, 2])
    plt.title('Coefficient path')
    plt.xlabel('L2 penalty ($\lambda$)')
    plt.ylabel('Coefficient value')
    plt.xscale('log')
    plt.rcParams.update({'font.size': 18})
    plt.tight_layout()
