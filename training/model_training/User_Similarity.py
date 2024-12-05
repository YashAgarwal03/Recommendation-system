from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

def user_similirity(matrix):
    user_item_sparse = csr_matrix(matrix)
    user_similarity = cosine_similarity(user_item_sparse)
    return user_similarity

