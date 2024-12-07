import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# Load dataset
dataset_path = 'C:\Recommendation-system\deployement\dataset_for_app.csv'
dataset = pd.read_csv(dataset_path)


# Functions from your code
def content_based(product_name, dataset=dataset, top_n=5):
    product_id = dataset[dataset['product_name'] == product_name]['product_id'].iloc[0]
    product_index = dataset[dataset['product_id'] == product_id].index[0]
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataset['combined_features'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    similarity_scores = list(enumerate(cosine_sim[product_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_indices = [i[0] for i in similarity_scores[1:top_n + 1]]
    return dataset.iloc[similar_indices][['product_name', 'category']]

def collaborative_filtering(user_id, dataset=dataset, top_n=5):
    dataset['rating'] = pd.to_numeric(dataset['rating'], errors='coerce')
    filtered_data = dataset[['user_id', 'product_id', 'rating']].dropna(subset=['rating'])
    user_item_matrix = filtered_data.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)
    user_item_sparse = csr_matrix(user_item_matrix)
    user_similarity = cosine_similarity(user_item_sparse)
    user_index = user_item_matrix.index.tolist().index(user_id)
    similar_users = user_similarity[user_index]
    similar_users_indices = similar_users.argsort()[::-1][1:top_n + 1]
    similar_users_items = user_item_matrix.iloc[similar_users_indices].mean(axis=0)
    recommended_items = similar_users_items.sort_values(ascending=False).head(top_n).index
    return dataset[dataset['product_id'].isin(recommended_items)][[ 'product_name', 'category']]

def hybrid_recommendation(user_id, product_name, dataset=dataset, top_n=5):
    # Find the product_id for the given product_name
    product_id = dataset[dataset['product_name'] == product_name]['product_id'].iloc[0]
    product_index = dataset[dataset['product_id'] == product_id].index[0]
    
    # Create the TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataset['combined_features'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Content-based filtering scores
    content_scores = list(enumerate(cosine_sim[product_index]))
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)
    content_recommendations = [dataset.iloc[x[0]]['product_id'] for x in content_scores[1:top_n + 1]]
    
    # Collaborative filtering
    dataset['rating'] = pd.to_numeric(dataset['rating'], errors='coerce')
    filtered_data = dataset[['user_id', 'product_id', 'rating']].dropna(subset=['rating'])
    user_item_matrix = filtered_data.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)
    user_item_sparse = csr_matrix(user_item_matrix)
    user_similarity = cosine_similarity(user_item_sparse)
    
    # Handle case where user_id may not exist in the dataset
    if user_id not in user_item_matrix.index:
        return pd.DataFrame(columns=['product_id', 'product_name', 'category'])
    
    user_index = user_item_matrix.index.tolist().index(user_id)
    similar_users = user_similarity[user_index]
    similar_users_indices = similar_users.argsort()[::-1][1:top_n + 1]
    similar_users_items = user_item_matrix.iloc[similar_users_indices].mean(axis=0)
    collaborative_recommendations = similar_users_items.sort_values(ascending=False).head(top_n).index.tolist()
    
    # Combine recommendations and remove duplicates
    hybrid_recommendations = list(set(content_recommendations + collaborative_recommendations))
    
    # Return the combined recommendations
    return dataset[dataset['product_id'].isin(hybrid_recommendations)][['product_name', 'category']]
