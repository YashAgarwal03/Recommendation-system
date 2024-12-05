import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print("Data successfully loaded.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

file_path = 'C:\Recommendation-system\data\dataset.csv'



if __name__ == "__main__":
    load_data(file_path)