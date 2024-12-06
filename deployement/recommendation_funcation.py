import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from dataset import dataset


def content_based(product_name,dataset=dataset,top_n = 5):
    # Find the product_id for the given product_name
    product_id = dataset[dataset['product_name'] == product_name]['product_id'].iloc[0]
    # Find the index where the product_id matches
    product_index = dataset[dataset['product_id'] == product_id].index[0]
    # Create the TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataset['combined_features'])
    # Compute cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    similarity_scores = list(enumerate(cosine_sim[product_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_indices = [i[0] for i in similarity_scores[1:top_n+1]]
    print(dataset.iloc[similar_indices][['product_id', 'product_name', 'category']])

def collaborative_filtering(user_id,dataset=dataset,top_n = 5):
    dataset['rating'] = pd.to_numeric(dataset['rating'], errors='coerce')
    # Prepare the data for user-item matrix
    filtered_data = dataset[['user_id', 'product_id', 'rating']].dropna(subset=['rating'])
    user_item_matrix = filtered_data.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)
    # Convert to sparse matrix
    user_item_sparse = csr_matrix(user_item_matrix)
    # Compute user-user cosine similarity
    user_similarity = cosine_similarity(user_item_sparse)
    user_index = user_item_matrix.index.tolist().index(user_id)
    similar_users = user_similarity[user_index]
    similar_users_indices = similar_users.argsort()[::-1][1:top_n+1]
    similar_users_items = user_item_matrix.iloc[similar_users_indices].mean(axis=0)
    recommended_items = similar_users_items.sort_values(ascending=False).head(top_n).index
    print(dataset[dataset['product_id'].isin(recommended_items)][['product_id', 'product_name', 'category']])

def hybird_recomend(user_id,product_name,dataset=dataset,top_n = 5):
    # Find the product_id for the given product_name
    product_id = dataset[dataset['product_name'] == product_name]['product_id'].iloc[0]
    # Find the index where the product_id matches
    product_index = dataset[dataset['product_id'] == product_id].index[0]
    # Create the TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataset['combined_features'])
    # Compute cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    dataset['rating'] = pd.to_numeric(dataset['rating'], errors='coerce')
    # Prepare the data for user-item matrix
    filtered_data = dataset[['user_id', 'product_id', 'rating']].dropna(subset=['rating'])
    user_item_matrix = filtered_data.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)
    # Convert to sparse matrix
    user_item_sparse = csr_matrix(user_item_matrix)
    # Compute user-user cosine similarity
    user_similarity = cosine_similarity(user_item_sparse)
    user_index = user_item_matrix.index.tolist().index(user_id)
    similar_users = user_similarity[user_index]
    
    content_scores = list(enumerate(cosine_sim[product_index]))
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)
    content_recommendations = [x[0] for x in content_scores[1:top_n+1]]

    # Collaborative filtering scores
    user_index = user_item_matrix.index.tolist().index(user_id)
    similar_users = user_similarity[user_index]
    similar_users_indices = similar_users.argsort()[::-1][1:top_n+1]
    similar_users_items = user_item_matrix.iloc[similar_users_indices].mean(axis=0)
    collaborative_recommendations = similar_users_items.sort_values(ascending=False).head(top_n).index.tolist()

    # Combine and remove duplicates
    hybrid_recommendations = list(set(content_recommendations + collaborative_recommendations))
    hybrid_recommendations = [i for i in hybrid_recommendations if isinstance(i, int)]
    print(dataset.iloc[hybrid_recommendations][['product_id', 'product_name', 'category']])


if __name__ == '__main__':
    hybird_recomend('UID01','Wayona Nylon Braided USB to Lightning Fast Charging and Data Sync Cable Compatible for iPhone 13, 12,11, X, 8, 7, 6, 5, iPad Air, Pro, Mini (3 FT Pack of 1, Grey)')