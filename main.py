import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score

dataset = pd.read_excel('/teamspace/studios/this_studio/jobs.xlsx')
datasetnull_count = dataset.isna().sum()
dataset1 = dataset.drop(['insights'],axis=1)
dataset1['title'] = dataset1['title'].astype(str)
dataset1['query'] = dataset1['query'].str.lower()

tfidf = TfidfVectorizer()
job_vecs = tfidf.fit_transform(dataset1['title'])

def get_recommendations(query, top_n=5):
    query_vec = tfidf.transform([query])
    cosine_scores = cosine_similarity(query_vec, job_vecs)
    top_scores_indices = np.argsort(-cosine_scores)[0][:top_n]
    top_scores = cosine_scores[0][top_scores_indices]
    recommendations = dataset1.iloc[top_scores_indices][['company', 'title', 'description', 'location', 'place', 'remote', 'type']]
    return recommendations, top_scores

def display_recommendations(query,recommendations, top_scores):
    st.write(f"Recommendations for your query:'{query}'")
    table_data = []
    for i, (idx, row) in enumerate(recommendations.iterrows()):
        match_type = "Perfect match" if top_scores[i] > 0.8 else "50% match" if top_scores[i] > 0.5 else "Not Match"
        match_explanation = f"The job title '{row['title']}' matches your query '{query}' based on the cosine similarity score of {top_scores[i]:.2f}"
        table_data.append([match_type, row['title'], match_explanation, row['company'], row['description'], f"{row['location']}, {row['place']}, {row['remote']}, {row['type']}"])
    table_df = pd.DataFrame(table_data, columns=["Match Type", "Job Title", "Match Explanation", "Company", "Description", "Location/Place/Remote/Type"])
    st.table(table_df)

def main():
    st.title("Job Recommendation Engine")
    query = st.text_input("Enter your job query:")
    if query:
        recommendations, top_scores = get_recommendations(query)
        display_recommendations(query,recommendations, top_scores)

if __name__ == "__main__":
    main()
