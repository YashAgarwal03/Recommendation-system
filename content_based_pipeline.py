
from training.model_training.Data_Ingestion import load_data,pd,file_path
from training.model_training.Data_Preprocessing import preprocess_data
from training.model_training.Feature_Engineering import feature_engineering
from training.model_training.Data_TfidfVectorizer import cosine_similirity
from training.model_training.Content_Based import recommend_content_based

def run_pipeline_cb(file_path:str)-> pd.DataFrame:
    # loading the csv file on the variable data
    data = load_data(file_path)

    # cleaning thr dataset 
    clean_data = preprocess_data(data)

    # making a new feature
    new_data = feature_engineering(clean_data)


    # cosine similirity matrix 
    cos_sim = cosine_similirity(new_data)


    # recommend the items 

    recommend = recommend_content_based(0,cos_sim,data)

    # return the recommend items 

    print(recommend)

if __name__ == '__main__':
    run_pipeline_cb(file_path)

     
