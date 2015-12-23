"""
k-Nearest Neighbours Regression
"""

# Imports
import numpy as np

# Functions
def get_data(data_frame, features, output):
    """
    Purpose: Extract features and prepare a feature matrix
             Set the first feature x0 = 1
    Input  : Original Dataframe, list of feature variables, output variable
    Output : Feature matrix array, output array
    """
    data_frame['constant'] = 1.0
    features = ['constant'] + features
    features_matrix = np.array(data_frame[features])
    if output != None:    
        output_array = np.array(data_frame[output])
    else:
        output_array = []   
    return(features_matrix, output_array.reshape((len(output_array))))

def normalize(features):
    """
    Purpose: Normalize feature matrix, each column of the matrix is a feature
    Input  : Unnormalized feature matrix
    Output : Normalized feature matrix, feature norms
    """
    norms = np.linalg.norm(features, axis=0)
    normalized_features = features/norms
    return (normalized_features, norms)

def compute_distances(features_instances, features_query):
    """
    Purpose: Compute distances between normalized training features
             and a query feature
    Input  : Normalized training features and normalized
             query features
    Output : Distances between the training and the query features
    """
    diff = features_instances - features_query
    distances = np.sqrt(np.sum(diff ** 2, axis=1))
    return(distances)
    
def k_nearest_neighbors(k, features_train, features_query):
    """
    Purpose: Determine the k nearest neighbours for one query
    Input  : Number of neighbours, normalized training features
             and query feature
    Output : 
    """
    distances = compute_distances(features_train, features_query)
    sorted_distances_idx = np.argsort(distances)
    neighbors = sorted_distances_idx[0:k]
    return(neighbors)
    
def predict_output_of_query(k, features_train, output_train, features_query):
    """
    Purpose: Predict the outputs for one query
    Input  : Number of neighbours, normalized training features,
             training output and normalized query features
    Output : Predictions for query features
    """
    k_nearest = k_nearest_neighbors(k, features_train, features_query)
    prediction = np.mean(output_train[k_nearest])
    return(prediction)

def predict(k, features_train, output_train, features_query):
    """
    Purpose: Predict the outputs for a multiple queries
    Input  : Number of neighbours, normalized training features,
             training output and normalized query features
    Output : Predictions for query features
    """
    predictions = np.zeros((features_query.shape[0]))
    for idx in range(features_query.shape[0]):
        predictions[idx] = predict_output_of_query(k, features_train, output_train, features_query[idx])
    return predictions

def get_residual_sum_of_squares(predictions, output):
    """
    Purpose: Compute residual sum of squares
    Input  : Predicted outputs and actual outputs
    Output : Residual sum of squares
    """
    residual = np.sum((predictions - output) ** 2)
    return(residual)
