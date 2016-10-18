import sframe
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from scipy.stats import multivariate_normal
from copy import deepcopy
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

# Function to load provided word counts
def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    return csr_matrix( (data, indices, indptr), shape)

# Create a sparse matrix from diagonals
def diag(array):
    n = len(array)
    return spdiags(array, 0, n, n)


def logpdf_diagonal_gaussian(x, mean, cov):
    '''
    Compute logpdf of a multivariate Gaussian distribution with diagonal covariance at a given point x.
    A multivariate Gaussian distribution with a diagonal covariance is equivalent
    to a collection of independent Gaussian random variables.

    x should be a sparse matrix. The logpdf will be computed for each row of x.
    mean and cov should be given as 1D numpy arrays
    mean[i] : mean of i-th variable
    cov[i] : variance of i-th variable
    '''
    n = x.shape[0]
    dim = x.shape[1]
    assert(dim == len(mean) and dim == len(cov))

    # multiply each i-th column of x by (1/(2*sigma_i)), where sigma_i is sqrt of variance of i-th variable.
    scaled_x = x.dot( diag(1./(2*np.sqrt(cov))) )
    # multiply each i-th entry of mean by (1/(2*sigma_i))
    scaled_mean = mean/(2*np.sqrt(cov))

    # sum of pairwise squared Eulidean distances gives SUM[(x_i - mean_i)^2/(2*sigma_i^2)]
    return -np.sum(np.log(np.sqrt(2*np.pi*cov))) - pairwise_distances(scaled_x, [scaled_mean], 'euclidean').flatten()**2

# Compute the log of a sum of exponentials
def log_sum_exp(x, axis):
    x_max = np.max(x, axis=axis)
    if axis == 1:
        return x_max + np.log( np.sum(np.exp(x-x_max[:,np.newaxis]), axis=1) )
    else:
        return x_max + np.log( np.sum(np.exp(x-x_max), axis=0) )


def EM_for_high_dimension(data, means, covs, weights, cov_smoothing=1e-5, maxiter=int(1e3), thresh=1e-4, verbose=False):
    # cov_smoothing: specifies the default variance assigned to absent features in a cluster.
    #                If we were to assign zero variances to absent features, we would be overconfient,
    #                as we hastily conclude that those featurese would NEVER appear in the cluster.
    #                We'd like to leave a little bit of possibility for absent features to show up later.
    n = data.shape[0]
    dim = data.shape[1]
    mu = deepcopy(means)
    Sigma = deepcopy(covs)
    K = len(mu)
    weights = np.array(weights)

    ll = None
    ll_trace = []

    for i in range(maxiter):
        # E-step: compute responsibilities
        logresp = np.zeros((n,K))
        for k in xrange(K):
            logresp[:,k] = np.log(weights[k]) + logpdf_diagonal_gaussian(data, mu[k], Sigma[k])
        ll_new = np.sum(log_sum_exp(logresp, axis=1))
        if verbose:
            print(ll_new)
        logresp -= np.vstack(log_sum_exp(logresp, axis=1))
        resp = np.exp(logresp)
        counts = np.sum(resp, axis=0)

        # M-step: update weights, means, covariances
        weights = counts / np.sum(counts)
        for k in range(K):
            mu[k] = (diag(resp[:,k]).dot(data)).sum(axis=0)/counts[k]
            mu[k] = mu[k].A1

            Sigma[k] = diag(resp[:,k]).dot( data.multiply(data)-2*data.dot(diag(mu[k])) ).sum(axis=0) \
                       + (mu[k]**2)*counts[k]
            Sigma[k] = Sigma[k].A1 / counts[k] + cov_smoothing*np.ones(dim)

        # check for convergence in log-likelihood
        ll_trace.append(ll_new)
        if ll is not None and (ll_new-ll) < thresh and ll_new > -np.inf:
            ll = ll_new
            break
        else:
            ll = ll_new

    out = {'weights':weights,'means':mu,'covs':Sigma,'loglik':ll_trace,'resp':resp}

    return out


def visualize_EM_clusters(tf_idf, means, covs, map_index_to_word):
    print('')
    print('==========================================================')

    num_clusters = len(means)
    for c in xrange(num_clusters):
        print('Cluster {0:d}: Largest mean parameters in cluster '.format(c))
        print('\n{0: <12}{1: <12}{2: <12}'.format('Word', 'Mean', 'Variance'))
        
        # The k'th element of sorted_word_ids should be the index of the word 
        # that has the k'th-largest value in the cluster mean. Hint: Use np.argsort().
        sorted_word_ids = np.argsort(means[c])[::-1][:len(means[c])]

        for i in sorted_word_ids[:5]:
            print '{0: <12}{1:<10.2e}{2:10.2e}'.format(map_index_to_word['category'][i], 
                                                       means[c][i],
                                                       covs[c][i])
        print '\n========================================================='


wiki = sframe.SFrame('../data/Week04/people_wiki.gl/').head(5000)
tf_idf = load_sparse_csr('../data/Week04/4_tf_idf.npz')  # NOT people_wiki_tf_idf.npz
map_index_to_word = sframe.SFrame('../data/Week04/4_map_index_to_word.gl/')  # NOT people_wiki_map_index_to_word.gl
tf_idf = normalize(tf_idf)

np.random.seed(5)
num_clusters = 25

# Use scikit-learn's k-means to simplify workflow
kmeans_model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=400, random_state=1, n_jobs=-1)
kmeans_model.fit(tf_idf)
centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_
means = [centroid for centroid in centroids]

# Initialize cluster weights
num_docs = tf_idf.shape[0]
weights = []
for i in xrange(num_clusters):
    # Compute the number of data points assigned to cluster i:
    num_assigned = len(cluster_assignment[cluster_assignment==i]) # YOUR CODE HERE
    w = float(num_assigned)/num_docs
    weights.append(w)

# Initialize cluster covariances
covs = []
for i in xrange(num_clusters):
    member_rows = tf_idf[cluster_assignment==i]
    cov = (member_rows.multiply(member_rows) - 2*member_rows.dot(diag(means[i]))).sum(axis=0).A1 / member_rows.shape[0] \
          + means[i]**2
    cov[cov < 1e-8] = 1e-8
    covs.append(cov)

# Run EM with initialization from k-means
out = EM_for_high_dimension(tf_idf, means, covs, weights, cov_smoothing=1e-10)
print out['loglik'] # print history of log-likelihood over time
visualize_EM_clusters(tf_idf, out['means'], out['covs'], map_index_to_word)

# Select random initialization
np.random.seed(5)
num_clusters = len(means)
num_docs, num_words = tf_idf.shape

random_means = []
random_covs = []
random_weights = []

for k in range(num_clusters):
    # Create a numpy array of length num_words with random normally distributed values.
    # Use the standard univariate normal distribution (mean 0, variance 1).
    # YOUR CODE HERE
    mean = np.random.randn(num_words)
    # Create a numpy array of length num_words with random values uniformly distributed between 1 and 5.
    # YOUR CODE HERE
    cov = np.random.uniform(1.0, 5.0, num_words)
    # Initially give each cluster equal weight.
    # YOUR CODE HERE
    weight = 1.0
    random_means.append(mean)
    random_covs.append(cov)
    random_weights.append(weight)

# Run EM with random initialization
out_random_init = EM_for_high_dimension(tf_idf, random_means, random_covs, random_weights, cov_smoothing=1e-5)
out_random_init['loglik'] > out['loglik']
visualize_EM_clusters(tf_idf, out_random_init['means'], out_random_init['covs'], map_index_to_word)
