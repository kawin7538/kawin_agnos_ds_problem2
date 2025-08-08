<!-- Slide number: 1 -->
# Symptom Recommender from Historical Data
Kawin Chinpong

<!-- Slide number: 2 -->
# Concept and Flow Overview
- Convert symptoms transactions from natural words into ICD10CM code, then finds association rules on ICD10CM code and answer back as natural words.
- Contains 2 phases
    - Data Preparation and Precomputed phase
    - Inference (prediction) Phase

<!-- Slide number: 3 -->
# Data Preparation and Precomputed Phase
- Select only ICD10CM codes (from External Dataset) that related with symptoms (R00-R99, 53x rows remains)
- Convert Description of ICD10CM into vector-based for similarity search (4096 dimensions using Qwen3-Embedding-8B)
- Convert search_term (from Internal Transaction Dataset) into list of string. For each symptom strings, convert into vector data.
- Calculate Cosine similarity for each vector-based symptoms compared with ICD10 Description, and get most appropriate ICD10CM from most similarity scores.
- Finally, Calculate frequent itemset using apriori (min_support = 1e-6 for more recall) then create association rules and export everything for Inference Phase.

<!-- Slide number: 4 -->
# Inference Phase
For each list of input query (eg., list_q = [“ปวดหัว”, “ตัวร้อน”], with max_items=10)
- Convert each words in list_q into vector-based and find similarity score for the most appropriated ICD10CM code.
- Apply these ICD10CM codes as LHS itemset, then find out top max_item related ICD10CM items from association_rule table ordered by most confidence score (with lift value >=1) (iterated through all RHS itemset and store until exceed criteria of max_items)
- For each related ICD10CM, convert it to natural words using a word that already found on Preparation Step.

<!-- Slide number: 5 -->
# Data Source
- Internal transaction dataset (only sample file is shown in this repo)
- ICD10CM codes and description (access from https://www.kaggle.com/datasets/mrhell/icd10cm-codeset-2023?resource=download)

<!-- Slide number: 6 -->
# Tech Stacks and Libraries
- Embedding model using Qwen/Qwen3-Embedding-8B from SentenceTransformer Library with Cosine Similarity Function.
- Apriori and Association Rules calculation using mlxtend library.
- API Interface using FastAPI with lru_cache

<!-- Slide number: 8 -->
# How to run this app
1. Environment and Library Initialization
    - For those who uses UV, run this command

        ```uv sync```

    - For those who uses pip, run this command

        ```pip install -r requirements.txt```

2. Run api python file (src/app_script.py) using this script

    ```python src/app_script.py```

<!-- Slide number: 9 -->
# Testing on Query API Endpoint
You may request this endpoint using Postman (or swaggerUI on FastAPI) by requests this Curl:

```curl -X POST "http://localhost:8012/query" -H "accept: application/json" -H "Content-Type: application/json" -d '{"list_q":["ปวดหัว"],"max_items":10}'```

Where payload must include these arguments:
- list_q: List of query or input that will be processed for recommendation, may be Thai of English.
- max_items: (Optional) Maximum amount of output items (default is 10)

Response of this endpoint is followed this structure:
- list_q: List of input string query, same as requested.
- list_recommendation: List of output string for next recommend symptoms, order by most confidence first.
- num_q: number of queries in list_q.
- num_recommendation: number of recommendations in list_recommendation.

<!-- Slide number: 11 -->
# Discussions, Limitations, and Suggestions
- Previously, I tried to apply LLM model to decide the appropriated ICD10CM code for a symptom based on top 20 similarity ICD10 codes (llama-3.1-8b-instant), but it got worse compared with this submitted version.
- For the next version, word lemmatization-like method (may be real lemmatization from words in Dictionary, or just another processes of LLM) to remove unnecessary words will be required to improve mapping processes (I also previously tried TH-EN translator on pythainlp, but got issues from the library instead, so I skipped in this version).