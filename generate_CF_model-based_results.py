#Import functions createdon my side
#from Functions_custo_by_Nico import * 
import pandas as pd
import os
import sys
import pickle
from joblib import dump, load
import gzip
import scipy.sparse
import implicit

#Build Data path
src_path = os.path.abspath(os.path.join("../IA_Project9_Openclassrooms_RecommandationContenu"))
if src_path not in sys.path:
    sys.path.append(src_path)
data_path = os.path.join(src_path, "dataset")

# Load train and test dataframes
df_clicks_metadata_train = pd.read_csv(os.path.join(data_path, "df_clicks_metadata_train.csv"))
df_clicks_metadata_test = pd.read_csv(os.path.join(data_path, "df_clicks_metadata_test.csv"))

# Load models
model_medium = load(os.path.join(data_path, "CF_model_based_medium_joblib_compressed.pkl")) #Compressed model using joblib compress=3

def compute_metrics_map_hitcount(y_true, y_pred, k=5):
    score = 0.0
    hit_count = 0.0
    for i, p in enumerate(y_pred):
        if p in y_true and p not in y_pred[:i]:
            hit_count += 1.0
            score += hit_count / (i+1.0)
    average_precision_k = score / min(len(y_true), k)
    return average_precision_k, hit_count

user_id_serie = df_clicks_metadata_test['user_id'].unique()
k=5

# Load previously saved Sparse Matrix
sparse_user_item = scipy.sparse.load_npz(os.path.join(data_path, 'CF_model_based_sparse_matrix_medium.npz'))

# Boucle for sur la liste des User_Id_Test, et mesure Top_k_accuracy, en utilisant les Embbeding réduit
score_serie = []
hit_count_serie = []
average_precision_k_serie = []
y_true_lenght_serie = []
y_pred_serie = []
y_true_serie = []
# Boucle for sur la liste des User_Id_Test, et mesure Top_k_accuracy, en utilisant les Embbeding réduit
for user in user_id_serie: 
    recommendations = model_medium.recommend(user, sparse_user_item[user], N=5, filter_already_liked_items=True)
    y_pred = recommendations[0].tolist()
    y_true = df_clicks_metadata_test.loc[df_clicks_metadata_test['user_id'] == user, 'click_article_id'].tolist()
    average_precision_k, hit_count = compute_metrics_map_hitcount(y_true, y_pred, k)
    # Append user recommendantions result in list
    hit_count_serie.append(hit_count)
    average_precision_k_serie.append(average_precision_k)
    y_true_lenght_serie.append(len(y_true))
    y_pred_serie.append(y_pred) #array_articles_embeddings_centré_réduit
    y_true_serie.append(y_true)

results_modelbased = pd.DataFrame(columns=['user_id', 'hit_count', 'average_precision', 'y_true_lenght_serie', 'y_pred_serie', 'y_true_serie'])
results_modelbased['user_id'] = user_id_serie
results_modelbased['hit_count'] = hit_count_serie
results_modelbased['average_precision'] = average_precision_k_serie
results_modelbased['y_true_lenght_serie'] = y_true_lenght_serie
results_modelbased['y_pred_serie'] = y_pred_serie
results_modelbased['y_true_serie'] = y_true_serie

Mean_Average_Precision_at_k = results_modelbased['average_precision'].mean() #np.mean([average_precision_k(y_true, y_pred, k=5) for y_true, y_pred in zip(y_true_serie, y_pred_serie)])
print(f"Mean_Average_Precision_at_k is = {Mean_Average_Precision_at_k}")
results_modelbased.to_csv(os.path.join(data_path, "results_modelbased_ALS_medium_normalized.csv"))