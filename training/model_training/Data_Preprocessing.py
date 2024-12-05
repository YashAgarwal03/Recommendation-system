import pandas as pd

def preprocess_data(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset["rating_count"].fillna(dataset["rating_count"].mode()[0], inplace = True)
    dataset['discounted_price'] = dataset['discounted_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)
    dataset['actual_price'] = dataset['actual_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)
    dataset['discount_percentage'] = dataset['discount_percentage'].replace('%', '', regex=True).astype(float)
    dataset['rating_count'] = dataset['rating_count'].replace({',': ''}, regex=True).astype(int)
    dataset['rating'] = pd.to_numeric(dataset['rating'], errors='coerce')
    dataset['discount_amount'] = dataset['actual_price'] - dataset['discounted_price']
    preprocessed_dataset = dataset
    return preprocessed_dataset
