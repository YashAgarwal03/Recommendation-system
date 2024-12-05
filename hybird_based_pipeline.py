
from training.model_training.Data_Ingestion import load_data,pd,file_path
from training.model_training.Data_Preprocessing import preprocess_data
from training.model_training.Feature_Engineering import feature_engineering
from training.model_training.Data_TfidfVectorizer import cosine_similirity
from training.model_training.User_Item_Matrix import user_item_matrix
from training.model_training.User_Similarity import user_similirity
from training.model_training.Collaborative_based import example_user_id
from training.model_training.Hybird import recommend_hybrid


def run_pipeline_hb(file_path:str)-> pd.DataFrame:
    # loading the csv file on the variable data
    data = load_data(file_path)

    # cleaning thr dataset 
    clean_data = preprocess_data(data)

    # making a new feature
    new_data = feature_engineering(clean_data)

    # cosine similirity matrix 
    cos_sim = cosine_similirity(new_data)

    # making a user_item_matrix
    matrix = user_item_matrix(clean_data)


    # user_similirity
    user_sim = user_similirity(matrix)


    # recommend the items 

    recommend = recommend_hybrid(0,example_user_id,cos_sim,user_sim,matrix,data)
    # return the recommend items 

    print(recommend)

if __name__ == '__main__':
    run_pipeline_hb(file_path)

     
