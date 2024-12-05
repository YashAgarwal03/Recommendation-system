
example_user_id ='AG3D6O4STAQKAY2UVGEUV46KN35Q,AHMY5CWJMMK5BJRBBSNLYT3ONILA,AHCTC6ULH4XB6YHDY6PCH2R772LQ,AGYHHIERNXKA6P5T7CZLXKVPT7IQ,AG4OGOFWXJZTQ2HKYIOCOY3KXF2Q,AENGU523SXMOS7JPDTW52PNNVWGQ,AEQJHCVTNINBS4FKTBGQRQTGTE5Q,AFC3FFC5PKFF5PMA52S3VCHOZ5FQ'

def recommend_cf_alternative(user_id, user_similarity_matrix, user_item_matrix,data, top_n=5):
    user_index = user_item_matrix.index.tolist().index(user_id)
    similar_users = user_similarity_matrix[user_index]
    similar_users_indices = similar_users.argsort()[::-1][1:top_n+1]
    similar_users_items = user_item_matrix.iloc[similar_users_indices].mean(axis=0)
    recommended_items = similar_users_items.sort_values(ascending=False).head(top_n).index
    return data[data['product_id'].isin(recommended_items)][['product_id', 'product_name', 'category']]