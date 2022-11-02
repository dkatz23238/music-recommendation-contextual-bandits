import pandas as pd
import re
import json
import pandas as pd
import json
import numpy as np
import random
from time import perf_counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import randomized_svd


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[\W_]+', ' ', text)
    return text


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# Top k artists
k = 1500


def get_user_artist_list(U_pp: pd.DataFrame, user_id: int):
    return [c for c in U_pp if U_pp.loc[user_id][c]]


A_v = (pd.read_csv(
    "./processed_data/artist_vectors.csv", encoding="utf-8").set_index("name")
    .astype(np.int8)
)

# Get top artists
top_k_artists = A_v.sum(1).sort_values().index[-k:]

A_v = A_v.loc[top_k_artists]
artists_texts = []

for artist in A_v.index:
    file = artist.replace(" ", "_").replace("/", "_").lower()
    try:
        with open(f"artists-json/{file}.json") as f:
            data = json.loads(f.read())

            artists_texts.append(data["text"])
    except FileNotFoundError:
        artists_texts.append("")


vectorizer = TfidfVectorizer(preprocessor=preprocess_text,
                             min_df=0.01,
                             max_df=0.5, token_pattern="\w+", strip_accents=None, ngram_range=(1, 1))

X = vectorizer.fit_transform(artists_texts)
df = pd.DataFrame(
    X.todense(), columns=vectorizer.get_feature_names_out(), index=A_v.index)


U, D, VT = randomized_svd(df, 100)

Udf = pd.DataFrame(U, index=top_k_artists)
Udf.index.name = "artist"
Udf.to_csv("Udf.csv")
