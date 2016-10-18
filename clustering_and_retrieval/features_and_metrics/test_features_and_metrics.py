# Imports
import sframe
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances

# Function to load provided word counts
def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']    
    return csr_matrix( (data, indices, indptr), shape)

# Create dictionary from word and word count using word index
def unpack_dict(matrix, map_index_to_word):
    table = list(map_index_to_word.sort('index')['category'])
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr
    num_doc = matrix.shape[0]
    return [{k:v for k,v in zip([table[word_id] for word_id in indices[indptr[i]:indptr[i+1]] ],
                                 data[indptr[i]:indptr[i+1]].tolist())} \
               for i in xrange(num_doc) ]

# Top words based on word counts
def top_words(name):
    row = wiki[wiki['name'] == name]
    word_count_table = row[['word_count']].stack('word_count', new_column_name=['word','count'])
    return word_count_table.sort('count', ascending=False)

# Top words based on tf-idf
def top_words_tf_idf(name):
    row = wiki[wiki['name'] == name]
    word_count_table = row[['tf_idf']].stack('tf_idf', new_column_name=['word','weight'])
    return word_count_table.sort('weight', ascending=False)

# Check is top words are present in the dictionary
def has_top_words(word_count_vector):
    unique_words = set(word_count_vector.keys())
    return common_words.issubset(unique_words)

# Read data
wiki = sframe.SFrame('../data/Week02/people_wiki.gl/')
wiki = wiki.add_row_number()
word_count = load_sparse_csr('../data/Week02/people_wiki_word_count.npz')
map_index_to_word = sframe.SFrame('../data/Week02/people_wiki_map_index_to_word.gl/')
tf_idf = load_sparse_csr('../data/Week02/people_wiki_tf_idf.npz')

# Inspect data on Barack Obama
wiki[wiki['name'] == 'Barack Obama']

# Find out index for Obama, Bush and Biden
wiki[wiki['name'] == 'Barack Obama'] #35817
wiki[wiki['name'] == 'George W. Bush'] #28447
wiki[wiki['name'] == 'Joe Biden'] #24478

# Nearest neighbour model based on word counts and 10 nearest neighbours
wiki['word_count'] = unpack_dict(word_count, map_index_to_word)
model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)
distances, indices = model.kneighbors(word_count[35817], n_neighbors=10)
neighbors = sframe.SFrame({'distance':distances.flatten(), 'id':indices.flatten()})
print wiki.join(neighbors, on='id').sort('distance')[['id','name','distance']]

obama_words = top_words('Barack Obama')
print obama_words
barrio_words = top_words('Francisco Barrio')
print barrio_words

combined_words = obama_words.join(barrio_words, on='word')
combined_words = combined_words.rename({'count':'Obama', 'count.1':'Barrio'})
combined_words.sort('Obama', ascending=False)

common_words = ['the', 'in', 'and', 'of', 'to']
common_words = set(['the', 'in', 'and', 'of', 'to'])
wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)
wiki_true = wiki[wiki['has_top_words'] == True]
wiki_false = wiki[wiki['has_top_words'] == False]
print wiki_true.num_rows()
print wiki_false.num_rows()

dist_obama_bush = euclidean_distances(word_count[35817], word_count[28447])
dist_bush_biden = euclidean_distances(word_count[28447], word_count[24478])
dist_biden_obama = euclidean_distances(word_count[24478], word_count[35817])

obama_words = top_words('Barack Obama')
print obama_words
bush_words = top_words('George W. Bush')
print bush_words

combined_words = obama_words.join(bush_words, on='word')
combined_words = combined_words.rename({'count':'Obama', 'count.1':'Bush'})
print combined_words.sort('Obama', ascending=False)

# Nearest neighbour model based on tf-idf and 10 nearest neighbours
wiki['tf_idf'] = unpack_dict(tf_idf, map_index_to_word)
model_tf_idf = NearestNeighbors(metric='euclidean', algorithm='brute')
model_tf_idf.fit(tf_idf)
distances, indices = model_tf_idf.kneighbors(tf_idf[35817], n_neighbors=10)
neighbors = sframe.SFrame({'distance':distances.flatten(), 'id':indices.flatten()})
print wiki.join(neighbors, on='id').sort('distance')[['id', 'name', 'distance']]

obama_tf_idf = top_words_tf_idf('Barack Obama')
print obama_tf_idf
schiliro_tf_idf = top_words_tf_idf('Phil Schiliro')
print schiliro_tf_idf

combined_words_tf_idf = obama_tf_idf.join(schiliro_tf_idf, on='word')
combined_words_tf_idf = combined_words_tf_idf.rename({'weight':'Obama', 'weight.1':'Schiliro'})
combined_words_tf_idf.sort('Obama', ascending=False)

common_words = ['obama', 'law', 'democratic', 'senate', 'presidential']
common_words = set(common_words)
wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)
wiki_true = wiki[wiki['has_top_words'] == True]
wiki_false = wiki[wiki['has_top_words'] == False]
print wiki_true.num_rows()
print wiki_false.num_rows()

dist_obama_bush_tf_idf = euclidean_distances(tf_idf[35817], tf_idf[28447])
dist_bush_biden_tf_idf = euclidean_distances(tf_idf[28447], tf_idf[24478])
dist_biden_obama_tf_idf = euclidean_distances(tf_idf[24478], tf_idf[35817])
