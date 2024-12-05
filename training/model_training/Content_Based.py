import pandas as pd

def recommend_content_based(product_index, cosine_sim_matrix, data, top_n=5) -> pd.DataFrame:
    similarity_scores = list(enumerate(cosine_sim_matrix[product_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_indices = [i[0] for i in similarity_scores[1:top_n+1]]
    return data.iloc[similar_indices][['product_id', 'product_name', 'category']]