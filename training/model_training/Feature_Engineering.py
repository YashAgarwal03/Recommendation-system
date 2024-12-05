import pandas as pd

def feature_engineering(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset['combined_features'] = (
    dataset['product_name'].fillna('') + ' ' +
    dataset['category'].fillna('') + ' ' +
    dataset['about_product'].fillna('') + ' ' +
    dataset['review_title'].fillna('') + ' ' +
    dataset['review_content'].fillna(''))
    new_dataset = dataset[['product_id', 'product_name', 'category','user_id', 'rating','combined_features']]
    return new_dataset