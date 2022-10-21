import streamlit as st
import requests
import pandas as pd
import os

# %% [markdown]
# # Création de l'Application Web
# L'idée est de produire une page web permettant le lancement du moteur de recommandation, via ppel à l'API Azure Function (via Streamlit).
# 
#     Pour lancer l'application Streamlit:
# **streamlit hello**
#     
#     Pour lancer la page Web Streamlit
# **streamlit run C:\Users\blanc\OpenClassrooms\IA_Project9_Openclassrooms_RecommandationContenu\streamlit\P9_01_streamlit.py**
# 

#Build Data path
src_path = os.getcwd()
data_path = os.path.join(src_path, "streamlit", "data") 
# Load train and test dataframes
df_clicks_metadata_test = pd.read_csv(os.path.join(data_path, "df_clicks_metadata_test.csv"), usecols=['user_id'])
user_id_list = df_clicks_metadata_test['user_id'].unique()


#Azure function url and endpoints
url = 'https://iap9openclassrooms.azurewebsites.net' 
endpoint = '/api'
http_trigger = '/article_reco_http_request'
get_variable = '?user_id='


# %%
st.title('IA Project 9 - My Content - Recommander system')
st.subheader("This app allows you to select user_id, and recommend 5 articles")
st.write('Recommender system is trained on Collaborative Filtering model based, using Implicit ALS')

# %%
st.subheader("1. Select User_ID for whom you want to recommend articles")
user_id = st.selectbox(f'user_id_list', user_id_list)
st.write(f'user_id selected = {user_id}, from user_id_list type : {type(user_id_list)}, with shape : {user_id_list.shape}')

# %%
st.subheader("2. Display Top5 Recommende articles")
azure_function_http_request_url = url+endpoint+http_trigger+get_variable+str(user_id)
st.write(f'API Backend endpoint = {azure_function_http_request_url}')
if st.button('Run Recommender system API call'):
    st.text(f"Launching azure function API using User_Id: {user_id}") 
    recommended_articles = requests.get(azure_function_http_request_url)
    st.write(f'Top5 Recommended articles for user_id {user_id}, are : {recommended_articles.text}')