from collections import defaultdict
import pickle as pkl
import pandas as pd
from sentence_transformers import SentenceTransformer, SimilarityFunction
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from tqdm import tqdm

icd_df = pd.read_csv("src/data/raw/ICDCodeSet.csv")
icd_df['ICDCode']=icd_df['ICDCode'].str.strip()
icd_df['Description']=icd_df['Description'].str.strip().str.lower()

# Filter on ICD10 on symptoms, injury, poisoning, and other causes (~4x,xxx rows)
# df=df[df['ICDCode'].str.contains("^(R|S|T)", regex=True)].reset_index(drop=True)

# Filter on ICD10 symptoms that can be found without clinical and lab diag (R00-R69) (517 rows)
# icd_df=icd_df[icd_df['ICDCode'].str.contains("^(R(0|1|2|3|4|5|6))", regex=True)].reset_index(drop=True)

# Filter on ICD10 symptoms (R00-R99) (567 rows)
icd_df=icd_df[icd_df['ICDCode'].str.contains("^(R)", regex=True)].reset_index(drop=True)

# Part 1: Encode specific ICD10 Description in vector-based for similarity search

# embedding with "Qwen/Qwen3-Embedding-8B", 4096 vector-dimensions
embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", similarity_fn_name=SimilarityFunction.COSINE)

documents = icd_df['Description'].values

document_embeddings = embedding_model.encode(documents, show_progress_bar=True, batch_size=32)

print(document_embeddings.shape)

icd_df.to_parquet("src/data/processed/icd10_document_metadata.parquet", index=False)

with open("src/data/processed/icd10_document_embeddings.pkl", "wb") as ff:
    pkl.dump(document_embeddings, ff)

symptom_transaction_df = pd.read_csv("src/data/raw/[CONFIDENTIAL] AI symptom picker data (Agnos candidate assignment) - ai_symptom_picker.csv", usecols=['search_term'])

symptom_transaction_df['patient_idx']=range(symptom_transaction_df.shape[0])

symptom_transaction_df['search_term']=symptom_transaction_df['search_term'].str.strip()
symptom_transaction_df['search_term']=symptom_transaction_df['search_term'].str.split(',').apply(lambda list_x: [x.strip() for x in list_x if len(x.strip())>0])

list_all_symptom_from_transaction = list(set(symptom_transaction_df['search_term'].explode().tolist()))
print(f"{len(list_all_symptom_from_transaction)=}")
query_embeddings = embedding_model.encode(list_all_symptom_from_transaction, prompt_name="query", show_progress_bar=True, batch_size=32)
similarity = embedding_model.similarity(query_embeddings, document_embeddings)

dict_master_word_icd10 = dict(zip(list_all_symptom_from_transaction, icd_df['ICDCode'].loc[similarity.argmax(dim=1)]))
dict_icd10_master_word = defaultdict(list)
for idx, val in tqdm(dict_master_word_icd10.items()):
    dict_icd10_master_word[val].append(idx)

with open("src/data/processed/dict_master_word_icd10.pkl", "wb") as ff:
    pkl.dump(dict_master_word_icd10, ff)
with open("src/data/processed/dict_icd10_master_word.pkl", "wb") as ff:
    pkl.dump(dict_icd10_master_word, ff)

symptom_transaction_df['icd10_term'] = symptom_transaction_df['search_term'].map(lambda list_x: list(dict.fromkeys([dict_master_word_icd10.get(x) for x in list_x])))
symptom_transaction_df

# Part 3: Convert ICD10 terms into association rules using Apriori algorithm

transformation_encoder_obj = TransactionEncoder()
icd10_terms_encoded = transformation_encoder_obj.fit_transform(symptom_transaction_df['icd10_term'])
icd10_terms_encoded = pd.DataFrame(icd10_terms_encoded, columns=transformation_encoder_obj.columns_)

frequent_itemsets = apriori(icd10_terms_encoded, min_support=1e-6, use_colnames=True, max_len=10)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=1e-6)
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
rules['antecedents']=rules['antecedents'].apply(lambda x: set(x))
rules['consequents']=rules['consequents'].apply(lambda x: set(x))

rules.to_parquet("src/data/processed/association_rules.parquet", index=False)