import random
from readers import read_dfs
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

import os

exp_dir = "./processed_data/"

atp, uap, uf = read_dfs()

tags_counts = (atp.groupby("tagID")
               .count()[["artistID"]]
               .sort_values("artistID"))

admissable_tag_ids = tags_counts.iloc[-1000:, :].index

_atp = atp[atp["tagID"].isin(admissable_tag_ids)]

artist_vectors = (
    pd.get_dummies(
        _atp[["tagValue", "name"]],
        columns=["tagValue"])
    .groupby("name")
    .sum(numeric_only=True))


user_artist_counts = uap.pivot_table(
    index="userID", columns="name", values="weight", aggfunc="sum").fillna(0)

normalized_plays = (user_artist_counts /
                    user_artist_counts.values.sum(axis=1)[:, None])

popular_artists = uap.groupby("name").sum(numeric_only=True
                                          )["weight"].sort_values(ascending=True)/1000
user_play_pair = (uap.pivot_table(
    index="userID", columns="name", values="weight", aggfunc=sum).fillna(0) > 0).astype(int)

# Clean up string encodings
artist_vectors.index = artist_vectors.index.str.encode(
    "latin-1").str.decode("utf-8")

user_play_pair.columns = user_play_pair.columns.str.encode(
    "latin-1").str.decode("utf-8")

artist_vectors.to_csv(exp_dir + "artist_vectors.csv")
user_play_pair.to_csv(exp_dir + "user_play_pair.csv")
