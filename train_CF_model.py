#Import functions createdon my side
#from Functions_custo_by_Nico import * 
import pandas as pd
import os
import sys
import gzip
import pickle
from joblib import dump
import scipy.sparse
import implicit

#Build Data path
src_path = os.path.abspath(os.path.join("../IA_Project9_Openclassrooms_RecommandationContenu"))
if src_path not in sys.path:
    sys.path.append(src_path)
data_path = os.path.join(src_path, "dataset")

# Load merged dataframe (click_all + metadata)
df_clicks_metadata_train = pd.read_csv(os.path.join(data_path, "df_clicks_metadata_train.csv"))
df_clicks_metadata_train["session_start"] = pd.to_datetime(df_clicks_metadata_train["session_start"])
df_clicks_metadata_train["click_timestamp"] = pd.to_datetime(df_clicks_metadata_train["click_timestamp"])

# Generate RATING : Calculate number of click per article, per user on single session, considering session_size discretized
def ratio_clicks_session_size_per_article_session(df_clicks_metadata):
    serie_clicks_ratings_per_session = df_clicks_metadata[['user_id', 'article_id', 'session_id', 'session_size_discretized']].groupby(['user_id', 'article_id', 'session_id']).size()
    serie_session_size = df_clicks_metadata[['user_id', 'article_id', 'session_id', 'session_size_discretized']].groupby(['user_id', 'article_id', 'session_id']).sum()
    df_ratio_clicks_session_size_over_article_clicks= pd.concat([serie_clicks_ratings_per_session, serie_session_size], axis=1).reset_index()
    df_ratio_clicks_session_size_over_article_clicks.rename(columns={0: 'clicks_hits'}, inplace = True)
    df_ratio_clicks_session_size_over_article_clicks['rating'] = df_ratio_clicks_session_size_over_article_clicks['session_size_discretized'] / df_ratio_clicks_session_size_over_article_clicks['clicks_hits']
    return df_ratio_clicks_session_size_over_article_clicks

df_ratio_clicks_session_size_over_article_clicks_session_train = ratio_clicks_session_size_per_article_session(df_clicks_metadata_train)

# Prepare matrix on training set: sparse matrix of item/user/rating
sparse_user_item_medium = scipy.sparse.csr_matrix((df_ratio_clicks_session_size_over_article_clicks_session_train['rating'].astype(float),
                                                 (df_ratio_clicks_session_size_over_article_clicks_session_train['user_id'], 
                                                 df_ratio_clicks_session_size_over_article_clicks_session_train['article_id'])))
scipy.sparse.save_npz(os.path.join(data_path, 'CF_model_based_sparse_matrix_medium.npz'), sparse_user_item_medium)

# Train Model: Alternating Least Square (ALS) Matrix Factorization in Collaborative Filtering
print("Model Training phase started")
model_medium = implicit.als.AlternatingLeastSquares(factors=100,iterations=200,regularization=0.1)
model_medium.fit(sparse_user_item_medium)

pickle.dump(model_medium, gzip.open(os.path.join(data_path, 'CF_model_based_medium_gzip_compressed.pkl'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
dump(model_medium, os.path.join(data_path, 'CF_model_based_medium_joblib_compressed.pkl'), compress=3)
print(f"Model Training completed, and pickle file saved in directory {data_path}")
