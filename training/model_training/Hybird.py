


def recommend_hybrid(product_index, user_id, cosine_sim_matrix, user_similarity_matrix, user_item_matrix, data, top_n=5):
    # Content-based scores
    content_scores = list(enumerate(cosine_sim_matrix[product_index]))
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)
    content_recommendations = [x[0] for x in content_scores[1:top_n+1]]

    # Collaborative filtering scores
    user_index = user_item_matrix.index.tolist().index(user_id)
    similar_users = user_similarity_matrix[user_index]
    similar_users_indices = similar_users.argsort()[::-1][1:top_n+1]
    similar_users_items = user_item_matrix.iloc[similar_users_indices].mean(axis=0)
    collaborative_recommendations = similar_users_items.sort_values(ascending=False).head(top_n).index.tolist()

    # Combine and remove duplicates
    hybrid_recommendations = list(set(content_recommendations + collaborative_recommendations))
    hybrid_recommendations = [i for i in hybrid_recommendations if isinstance(i, int)]
    return data.iloc[hybrid_recommendations][['product_id', 'product_name', 'category']]