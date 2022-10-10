#Import functions createdon my side
from Functions_custo_by_Nico import * 
import pickle
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import top_k_accuracy_score
from scipy.spatial import distance
import scipy.sparse
import implicit
from surprise import accuracy, Reader, Dataset, SVD, NormalPredictor
from surprise.model_selection import KFold, PredefinedKFold, train_test_split, cross_validate, GridSearchCV

#Build Data path
src_path = os.path.abspath(os.path.join("../IA_Project9_Openclassrooms_RecommandationContenu"))
if src_path not in sys.path:
    sys.path.append(src_path)
data_path = os.path.join(src_path, "dataset")

# Load train and test dataframes
df_clicks_metadata_train = pd.read_csv(os.path.join(data_path, "df_clicks_metadata_train.csv"))
df_clicks_metadata_test = pd.read_csv(os.path.join(data_path, "df_clicks_metadata_test.csv"))
df_clicks_metadata = pd.read_csv(os.path.join(data_path, "df_clicks_metadata.csv"))
df_metadata = pd.read_csv(os.path.join(data_path, "articles_metadata.csv"))
df_metadata["created_at_ts"] = pd.to_datetime(df_metadata["created_at_ts"],unit="ms")

# Load models
model_easy_pickle = open(os.path.join(data_path, "CF_model_based_easy.pkl"),"rb")
model_easy = pickle.load(model_easy_pickle)
model_easy_pickle.close()
model_medium_pickle = open(os.path.join(data_path, "CF_model_based_medium.pkl"),"rb")
model_medium = pickle.load(model_medium_pickle)
model_medium_pickle.close()
model_complex_pickle = open(os.path.join(data_path, "CF_model_based_complex.pkl"),"rb")
model_complex = pickle.load(model_complex_pickle)
model_complex_pickle.close()


# open , load and close pickled file
articles_embeddings = open(os.path.join(data_path, "articles_embeddings.pickle"),"rb")
array_articles_embeddings = pickle.load(articles_embeddings)
articles_embeddings.close()
articles_embeddings_centré_réduit = open(os.path.join(data_path, "articles_embeddings_centré_réduit.pickle"),"rb")
array_articles_embeddings_centré_réduit = pickle.load(articles_embeddings_centré_réduit)
articles_embeddings_centré_réduit.close()

def recommended_top5_contentbased(array_articles_embeddings, userId, df_clicks_metadata):
    #get a list of all articles read by user
    var = df_clicks_metadata.loc[df_clicks_metadata['user_id']==userId]['click_article_id'].tolist()
    #Compute the Mean Embedding from all past articles
    array_past_articles_embeddings = np.zeros(shape=(len(var), (array_articles_embeddings.shape)[1]))
    for i in range(0, len(var)):
        value = var[i]
        array_past_articles_embeddings[i] = np.copy(array_articles_embeddings[value])
    array_past_articles_Mean_embeddings = np.mean(array_past_articles_embeddings, axis=0).tolist()

    #get 5 articles the most similar to the selected one, using embedding weights, looking 5 indexes from 0 to 5
    distances = distance.cdist([array_past_articles_Mean_embeddings], array_articles_embeddings, "cosine")[0]
    result = (np.argsort(distances)[0:5])
    j = 6
    for i in range(0,5):
        if (result[i] in var):
            while ((np.argsort(distances)[j]) in var):
                j = j + 1 
            else:
                result[i] = np.argsort(distances)[j]   
    return result

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

 # MEDIUM RATING : Calculate number of click per article, per user on single session, per article
def ratio_clicks_session_size_per_article_session(df_clicks_metadata):
    serie_clicks_ratings_per_session = df_clicks_metadata[['user_id', 'article_id', 'session_id', 'session_size_discretized']].groupby(['user_id', 'article_id', 'session_id']).size()
    serie_session_size = df_clicks_metadata[['user_id', 'article_id', 'session_id', 'session_size_discretized']].groupby(['user_id', 'article_id', 'session_id']).sum()
    df_ratio_clicks_session_size_over_article_clicks= pd.concat([serie_clicks_ratings_per_session, serie_session_size], axis=1).reset_index()
    df_ratio_clicks_session_size_over_article_clicks.rename(columns={0: 'clicks_hits'}, inplace = True)
    df_ratio_clicks_session_size_over_article_clicks['rating'] = df_ratio_clicks_session_size_over_article_clicks['session_size_discretized'] / df_ratio_clicks_session_size_over_article_clicks['clicks_hits']
    return df_ratio_clicks_session_size_over_article_clicks
    
df_ratio_clicks_session_size_over_article_clicks_session_train = ratio_clicks_session_size_per_article_session(df_clicks_metadata_train)
df_ratio_clicks_session_size_over_article_clicks_session_test = ratio_clicks_session_size_per_article_session(df_clicks_metadata_test)


sparse_user_item = scipy.sparse.csr_matrix((df_ratio_clicks_session_size_over_article_clicks_session_train['rating'].astype(float),
                                                 (df_ratio_clicks_session_size_over_article_clicks_session_train['user_id'], 
                                                 df_ratio_clicks_session_size_over_article_clicks_session_train['article_id'])))

# Boucle for sur la liste des User_Id_Test, et mesure Top_k_accuracy, en utilisant les Embbeding réduit
score_serie = []
hit_count_serie = []
average_precision_k_serie = []
y_true_lenght_serie = []
y_pred_serie = []
y_true_serie = []
# Boucle for sur la liste des User_Id_Test, et mesure Top_k_accuracy, en utilisant les Embbeding réduit
for user in user_id_serie: #df_clicks_metadata_test['user_id'].unique()
    recommendations = model_medium.recommend(user, sparse_user_item[user], N=5)
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