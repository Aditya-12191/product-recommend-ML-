import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    df = pd.read_csv('data/products.csv')
    return df

def build_recommender(df):
    tfidf = TfidfVectorizer(stop_words='english')
    df['description'] = df['description'].fillna('')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def recommend(product_name, df, cosine_sim):
    indices = pd.Series(df.index, index=df['product_name']).drop_duplicates()
    idx = indices.get(product_name)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]
    product_indices = [i[0] for i in sim_scores]
    return df['product_name'].iloc[product_indices].tolist()
