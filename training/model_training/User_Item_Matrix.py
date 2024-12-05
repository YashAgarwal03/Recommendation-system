

def user_item_matrix(data):
    filtered_data = data[['user_id', 'product_id', 'rating']].dropna(subset=['rating'])
    user_item_matrix = filtered_data.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)
    return user_item_matrix

