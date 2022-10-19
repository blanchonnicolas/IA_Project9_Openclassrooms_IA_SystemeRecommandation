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
- Main repo : [Notebooks and Scripts](https://github.com/blanchonnicolas/IA_Project9_Openclassrooms_IA_SystemeRecommandationp) 
- [Azure Function](https://github.com/blanchonnicolas/IA_Project9_Openclassrooms_IA_SystemeRecommandation/azure_function "Azure Function") : **this repo**
- [Mobile App](https://github.com/blanchonnicolas/IA_Project9_Openclassrooms_IA_SystemeRecommandation/streamlit "Mobile App")



