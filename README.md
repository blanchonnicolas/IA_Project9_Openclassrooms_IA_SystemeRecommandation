# Project "My Content"

Context: Recommender System
Goals: Solutions supporting user in article choice, by providing top 5 recommandations
Repository of OpenClassrooms project 9' [AI Engineer path](https://openclassrooms.com/fr/paths/188)

## My Content

Our role is to participate to the conception and development of the recommender system:
 - Item-Based (using article metadata and embeddings)
 - Collaborative filtering - Model based
 - Deploy solution on Cloud, using serverless platform *(Azure function)*

The project is using below dataset to train recommender systems: [My Content dataset](https://www.kaggle.com/gspmoreira/news-portal-user-interactions-by-globocom#clicks_sample.csv).

Finally, recommender system engine is callable using : [Azure function called with user_id 72 - Http trigger](https://iap9openclassrooms.azurewebsites.net/api/article_reco_http_request?user_id=72)

## How to use
For Setup:
- Python and VSCode with Jupyter extension to read notebooks.
For application access:
- Acces to web, through Streamlit-share app.

This repository is part of a 3-repos project :
- Main repo : [Notebooks and Scripts](https://github.com/blanchonnicolas/IA_Project9_Openclassrooms_IA_SystemeRecommandationp) : **this repo**
- [Azure Function](https://github.com/blanchonnicolas/IA_Project9_Openclassrooms_IA_SystemeRecommandation/tree/main/azure_function "Azure Function")
- [Mobile App](https://github.com/blanchonnicolas/IA_Project9_Openclassrooms_IA_SystemeRecommandation/tree/main/streamlit "Mobile App")


## More details here :

-   [Presentation](https://github.com/blanchonnicolas/IA_Project9_Openclassrooms_IA_SystemeRecommandation)

-   [Technical Note](https://github.com/blanchonnicolas/IA_Project9_Openclassrooms_IA_SystemeRecommandation)

-   [Vidéo]()

## Main Repo content
-   [Notebook 1 : datagenerator](https://github.com/blanchonnicolas/IA_Project9_Openclassrooms_IA_SystemeRecommandation/blob/main/P9_01_notebook.ipynb)
    - Data Analysis: Visualisation, Modification 
	- Unit Test

-   [Train Collaboraive Filtering model - script](https://github.com/blanchonnicolas/IA_Project9_Openclassrooms_IA_SystemeRecommandation/blob/main/train_CF_model.py)
    - Rating generation (Clicks_hint)
    - Creation of Scipy.sparse csr matrix (User/Article/Rating)
    - Training of Implicit ALS model (model.fit())
    - Joblib save using compression

-   [Compute CF Model Results - script](https://github.com/blanchonnicolas/IA_Project9_Openclassrooms_IA_SystemeRecommandation/blob/main/generate_CF_model-based_results.py)
    - Metrics generation (Clicks_hint, Precision@k, MeanAverage@k)
    - Test_User_id loop to recomend using **trained Implicit model**, and compare prediction with real
    - Measure and store results

-   [Compute Item based Results - script](https://github.com/blanchonnicolas/IA_Project9_Openclassrooms_IA_SystemeRecommandation/blob/main/generate_CF_model-based_results.py)
    - Metrics generation (Clicks_hint, Precision@k, MeanAverage@k)
    - Test_User_id loop to recomend using **COSINE SIMILARITY from Embedding**, and compare prediction with real
    - Measure and store results

