import logging
import azure.functions as func
import pandas as pd
from joblib import load
import scipy.sparse
import os
import implicit

# Build Data path
sys_path = os.getcwd()
src_path = os.path.join(sys_path, "data")
sparsematrix_path = os.path.join(src_path, "CF_model_based_sparse_matrix_medium.npz")
model_path = os.path.join(src_path, "CF_model_based_medium_joblib_compressed.pkl")

# Load model and sparsematrix
model_medium = load(model_path)
sparse_user_item = scipy.sparse.load_npz(sparsematrix_path)

# Function to run recommendation model
def get_reco(user_id):
    article_list = model_medium.recommend(user_id, sparse_user_item[user_id], N=5)
    return article_list[0].tolist()

# Function trigger http request
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    user_id = req.params.get('user_id')
    if not user_id:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            user_id = req_body.get('user_id')
    if user_id:
        # Run the function for recommendation (Collaborative filtering)
        article_list = get_reco(int(user_id))
        str_articles = ", ".join([str(x) for x in article_list])
        return func.HttpResponse(str_articles)
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a user_id in the query string or in the request body for a personalized response.",
             status_code=200
        )

