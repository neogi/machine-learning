# Imports
import string
import numpy as np

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
def compute_feature_derivative(errors, feature):
    """
    Purpose: Compute derivative of a feature wrt coefficient
    Input  : Error between true output and predicted output values, feature,
             coefficient, if the feature is constant or not
    Output : Derivative of the feature wrt coefficient
    """
    return feature.T.dot(errors)

# Function to compute log likelihood
def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    """
    Purpose: Compute Log-Likelihood
    Input  : Feature matrix, coefficients, true output values
    Output : Log-Likelihood
    """
    indicator = (sentiment == +1)
    scores = compute_score(feature_matrix, coefficients)
    log_likelihood = np.sum((indicator - 1) * scores - np.log(1.0 + np.exp(-scores)))
    return log_likelihood

# Function to compute average log likelihood
def compute_avg_log_likelihood(feature_matrix, sentiment, coefficients):
    """
    Purpose: Compute average Log-Likelihood
    Input  : Feature matrix, coefficients, true output values
    Output : Log-Likelihood
    """
    indicator = (sentiment == +1)
    scores = compute_score(feature_matrix, coefficients)
    logexp = np.log(1.0 + np.exp(-scores))
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    log_likelihood = np.sum((indicator - 1) * scores - logexp)/len(feature_matrix)
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
def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    """
    Purpose: Perform logistic regression
    Input  : Feature matrix, true output values, initial estimate of coefficients, step size
             maximum number of iterations
    Output : Estimated coefficient vector
    """
    coefficients = np.array(initial_coefficients)
     
    for itr in xrange(max_iter):
        predictions = compute_probability(compute_score(feature_matrix, coefficients))
        indicator = (sentiment == +1)*1.0
        errors = indicator - predictions
         
        for j in xrange(len(coefficients)):
            derivative = compute_feature_derivative(errors, feature_matrix[:, j])
            coefficients[j] = coefficients[j] + step_size * derivative
     
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
            (int(np.ceil(np.log10(max_iter))), itr, lp)
     
    return coefficients

# Function to perform logistic regression (stochastic gradient)
def logistic_regression_SG(feature_matrix, sentiment, initial_coefficients, step_size, batch_size, max_iter):
    """
    Purpose: Perform logistic regression using stochastic gradient ascent
    Input  : Feature matrix, true output values, initial estimate of coefficients, step size
             batch size, maximum number of iterations
    Output : Estimated coefficient vector, log likelihood
    """
    log_likelihood_all = []
    coefficients = np.array(initial_coefficients)
    np.random.seed(seed=1)
    permutation = np.random.permutation(len(feature_matrix))
    feature_matrix = feature_matrix[permutation,:]
    sentiment = sentiment[permutation]

    i = 0
    for itr in xrange(max_iter):
        predictions = compute_probability(compute_score(feature_matrix[i:i+batch_size,:], coefficients))
        indicator = (sentiment[i:i+batch_size] == +1)*1.0
        errors = indicator - predictions

        for j in xrange(len(coefficients)):
            derivative = compute_feature_derivative(errors, feature_matrix[i:i+batch_size,j])
            coefficients[j] = coefficients[j] + step_size * derivative/batch_size

        lp = compute_avg_log_likelihood(feature_matrix[i:i+batch_size,:], sentiment[i:i+batch_size], coefficients)
        log_likelihood_all.append(lp)

        if itr <= 15 or (itr <= 1000 and itr % 100 == 0) or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0 or itr == max_iter-1:
            data_size = len(feature_matrix)
            print 'Iteration %*d: Average log likelihood (of data points  [%0*d:%0*d]) = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, \
                 int(np.ceil(np.log10(data_size))), i, \
                 int(np.ceil(np.log10(data_size))), i+batch_size, lp)  

        i += batch_size
        if i+batch_size > len(feature_matrix):
            permutation = np.random.permutation(len(feature_matrix))
            feature_matrix = feature_matrix[permutation,:]
            sentiment = sentiment[permutation]
            i = 0                

    return coefficients, log_likelihood_all

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
