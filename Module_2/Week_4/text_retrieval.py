import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore

vi_data_df = pd.read_csv('data/vi_text_retrieval.csv')
context = vi_data_df['text']
context = [doc.lower() for doc in context]

tfidf_vectorizer = TfidfVectorizer()
context_embedded = tfidf_vectorizer.fit_transform(context)
print(context_embedded.toarray()[7][0])

def tfidf_search(question, tfidf_vectorizer, top_d=5):
    query_embedded = tfidf_vectorizer.transform([question.lower()])
    cosine_scores = cosine_similarity(query_embedded, 
                                      context_embedded).reshape((-1,))

    result = []
    for idx in cosine_scores.argsort()[-top_d:][::-1]:
        doc_score = {
            'id': idx,
            'cosine_score': cosine_scores[idx]
        }
        result.append(doc_score)

    return result

question = vi_data_df.iloc[0]['question']
result = tfidf_search(question, tfidf_vectorizer)
print(round(result[0]['cosine_score'], 2))

def corr_search(question, tfidf_vectorizer, top_d=5):
    query_embedded = tfidf_vectorizer.transform([question.lower()])
    corr_scores = np.corrcoef(query_embedded.toarray()[0],
                              context_embedded.toarray())
    corr_scores = corr_scores[0][1:]
    result = []
    for idx in corr_scores.argsort()[-top_d:][::-1]:
        doc = {
            'id': idx,
            'corr_score': corr_scores[idx]
        }
        result.append(doc)

    return result

question = vi_data_df.iloc[0]['question']
result = corr_search(question, tfidf_vectorizer, top_d=5)
print(round(result[1]['corr_score'], 2))