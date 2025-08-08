from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel
from functools import cache, lru_cache
import pickle as pkl
import pandas as pd
from sentence_transformers import SentenceTransformer, SimilarityFunction
import uvicorn
from fastapi import FastAPI

class QueryInputSchema(BaseModel):
    list_q: List[str]
    max_items: Optional[int] = 10

class QueryOutputSchema(BaseModel):
    list_q: List[str]
    list_recommendation: List[str]
    num_q: int
    num_recommendation: int

# embedding with "Qwen/Qwen3-Embedding-8B", 4096 vector-dimensions
embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", similarity_fn_name=SimilarityFunction.COSINE)

icd10_document_metadata = pd.read_parquet("src/data/processed/icd10_document_metadata.parquet")
with open("src/data/processed/icd10_document_embeddings.pkl", "rb") as ff:
    document_embeddings = pkl.load(ff)
with open("src/data/processed/dict_master_word_icd10.pkl", "rb") as ff:
    dict_master_word_icd10 = pkl.load(ff)
with open("src/data/processed/dict_icd10_master_word.pkl", "rb") as ff:
    dict_icd10_master_word = pkl.load(ff)
association_rule_df = pd.read_parquet("src/data/processed/association_rules.parquet")

@lru_cache
def get_symptom_recommendation(tuple_q: Tuple[str], max_items: int = 10):

    list_q = list(tuple_q)
    
    # convert list of queries into list of ICD10 using cosine similarity

    temp_list_q = [q.strip().lower() for q in list_q]
    print(f"{temp_list_q=}")
    query_embeddings = embedding_model.encode(temp_list_q, prompt_name="query", show_progress_bar=True, batch_size=32)
    similarity = embedding_model.similarity(query_embeddings, document_embeddings)

    temp_list_icd10 = icd10_document_metadata['ICDCode'].loc[similarity.argmax(dim=1)].tolist()
    # temp_list_icd10 = list(dict.fromkeys(temp_list_icd10))
    temp_list_icd10 = list(set(temp_list_icd10))

    print(f"{temp_list_icd10=}")

    # find recommendations from association rule df

    masked = association_rule_df['antecedents'].apply(lambda x: set(x)==set(temp_list_icd10))

    ans_df = association_rule_df[masked].sort_values('confidence', ascending=False)
    ans_df = ans_df[(ans_df['lift']>=1)]
    print(ans_df.head(20))

    list_icd10_answer = list()
    for list_consequents in ans_df['consequents'].values:
        
        list_icd10_answer.extend(list_consequents)
        list_icd10_answer = [item for item in list_icd10_answer if item not in temp_list_icd10]
        # list_icd10_answer = list(dict.fromkeys(list_icd10_answer))
        list_icd10_answer = list(set(list_icd10_answer))

        if len(list_icd10_answer)>max_items:
            break;
    
    # find recommendations from association from chief complain if list of answer is less than desired numbers
    if len(list_icd10_answer)<max_items and len(temp_list_q)>1:
        masked = association_rule_df['antecedents'].apply(lambda x: set(x)==set([temp_list_icd10[0]]))
        ans_df = association_rule_df[masked].sort_values('confidence', ascending=False)
        ans_df = ans_df[(ans_df['lift']>=1)]
        for list_consequents in ans_df['consequents'].values:
        
            list_icd10_answer.extend(list_consequents)
            list_icd10_answer = [item for item in list_icd10_answer if item not in temp_list_icd10]
            # list_icd10_answer = list(dict.fromkeys(list_icd10_answer))
            list_icd10_answer = list(set(list_icd10_answer))

            if len(list_icd10_answer)>max_items:
                break;
    
    print(f"{list_icd10_answer=}")

    list_answer = [dict_icd10_master_word.get(item, list())[0].capitalize() for item in list_icd10_answer if len(dict_icd10_master_word.get(item, list()))>0][:max_items]
    print(f"IN FUNCTION {list_answer=}")

    return list_answer

app = FastAPI(title="Kawin Agnos Symptom Recommender System", version="1.0.0")

@app.get("/healthz")
def healthz():
    return {"status": "success"}

@app.post("/query")
def query(body: QueryInputSchema) -> QueryOutputSchema:
    print(f"list_q = {body.list_q}")
    answer = get_symptom_recommendation(tuple(body.list_q), body.max_items)
    print(f"{answer=}")
    return {
        "list_q": body.list_q,
        "list_recommendation": answer,
        "num_q": len(body.list_q),
        "num_recommendation": len(answer)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8012)