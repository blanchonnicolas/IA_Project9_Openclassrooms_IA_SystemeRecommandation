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

# Boucle for sur la liste des User_Id_Test, et mesure Top_k_accuracy, en utilisant les Embbeding réduit
score_serie = []
hit_count_serie = []
average_precision_k_serie = []
y_true_lenght_serie = []
y_pred_serie = []
y_true_serie = []
for user in user_id_serie: #df_clicks_metadata_test['user_id'].unique()
    y_pred = recommended_top5_contentbased(array_articles_embeddings, user, df_clicks_metadata_train).tolist() #array_articles_embeddings_centré_réduit
    y_true = df_clicks_metadata_test.loc[df_clicks_metadata_test['user_id'] == user, 'click_article_id'].tolist()
    average_precision_k, hit_count = compute_metrics_map_hitcount(y_true, y_pred, k)
    # Append user recommendantions result in list
    hit_count_serie.append(hit_count)
    average_precision_k_serie.append(average_precision_k)
    y_true_lenght_serie.append(len(y_true))
    y_pred_serie.append(y_pred) #array_articles_embeddings_centré_réduit
    y_true_serie.append(y_true)
results_contentbased = pd.DataFrame(columns=['user_id', 'hit_count', 'average_precision', 'y_true_lenght_serie', 'y_pred_serie', 'y_true_serie'])
results_contentbased['user_id'] = user_id_serie
results_contentbased['hit_count'] = hit_count_serie
results_contentbased['average_precision'] = average_precision_k_serie
results_contentbased['y_true_lenght_serie'] = y_true_lenght_serie
results_contentbased['y_pred_serie'] = y_pred_serie
results_contentbased['y_true_serie'] = y_true_serie

# array_results_contentbased = np.empty((0,6), dtype=object)
# for user in user_id_serie: #df_clicks_metadata_test['user_id'].unique()
#     y_pred = recommended_top5_contentbased(array_articles_embeddings_centré_réduit, user, df_clicks_metadata_train).tolist() #array_articles_embeddings_centré_réduit
#     y_true = df_clicks_metadata_test.loc[df_clicks_metadata_test['user_id'] == user, 'click_article_id'].tolist()
#     average_precision_k, hit_count = compute_metrics_map_hitcount(y_true, y_pred, k)
#     # Append user recommendantions result in list
#     array_results_contentbased = np.append(array_results_contentbased, np.array([[user, hit_count, average_precision_k, len(y_true), np.array(y_pred, dtype=object), np.array(y_true, dtype=object)]]), axis=0)
# results_contentbased = pd.DataFrame(array_results_contentbased, columns=['user_id', 'hit_count', 'average_precision', 'y_true_lenght', 'y_pred', 'y_true'])

results_contentbased.to_csv(os.path.join(data_path, "results_contentbased_embedding_full.csv"))