import pandas as pd
import json
import numpy as np
import random
from time import perf_counter
import os
from sklearn.utils.extmath import randomized_svd

if os.path.exists(".linucb"):
    with open(".linucb") as f:
        text = f.read()
        rounds = int(text)+1
    with open(".linucb", "w") as f:
        f.write(str(rounds))

else:
    rounds = 0
    with open(".linucb", "w") as f:
        f.write(str(rounds))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


use_wiki_data = False
use_only_wiki_data = False
center_data = False
binarize_tags = False
svd_tags = True
svd_tags_components = 20

# Top k artists
k = 1000
# J experiments
J = 150
# H trials
H = 35


def get_user_artist_list(U_pp: pd.DataFrame, user_id: int):
    return [c for c in U_pp if U_pp.loc[user_id][c]]


A_v = (pd.read_csv(
    "./processed_data/artist_vectors.csv", encoding="utf-8").set_index("name")
    .astype(np.int8)
).apply(np.log1p)

original_Av_columns = A_v.columns

if binarize_tags:
    A_v = (A_v > 0).astype(int)


U_pp = (pd.read_csv(
    "./processed_data/user_play_pair.csv", encoding="utf-8").set_index("userID")
    .astype(np.int8)
)

Udf = pd.read_csv("Udf.csv").set_index("artist")

# Get top artists
top_k_artists = A_v.sum(1).sort_values().index[-k:]

A_v = A_v.loc[top_k_artists]
artists = A_v.index
U_pp = U_pp[artists]


user_ids = U_pp.index.tolist()
random.shuffle(user_ids)

# Reg parameter
alpha = 1
experiment_data = {}
experiment_data["use_wiki_data"] = use_wiki_data
experiment_data["use_only_wiki_data"] = use_only_wiki_data
experiment_data["svd_tags"] = svd_tags
experiment_data["center_data"] = center_data
experiment_data["svd_tags_components"] = svd_tags_components
experiment_data["binarize_tags"] = binarize_tags

if center_data:
    # Center design matrix
    A_v = A_v - A_v.mean()


if svd_tags:
    U, D, VT = randomized_svd(A_v, svd_tags_components)
    A_v = pd.DataFrame(U, index=A_v.index)

if use_wiki_data and not use_only_wiki_data:
    A_v = A_v.merge(Udf, right_index=True, left_index=True)

if use_only_wiki_data:
    A_v = Udf

for uid in user_ids[:J]:

    X = A_v
    n, m = A_v.shape
    # Convert 0,1 range to -1,1 range
    target = U_pp.loc[uid].apply(lambda x: 1 if x else -1)

    A = np.identity(m)
    b = np.zeros(m)

    seen = []
    decision_sequence = []

    accumulated_rewards = 0
    control_rewards = 0
    experiment_data[uid] = {}

    for t in range(H):

        A_inv = np.linalg.inv(A)

        theta = A_inv @ b

        mus = X.values.dot(theta)

        # X.T @ A_inv @ X
        std = alpha * np.sqrt(
            np.diag(X.dot(A_inv).values @ X.T)
        )

        # Upper confidence bound of reward
        artist_scores = mus + std
        random_artist = random.choice(X.index)
        scores_argsort = np.argsort(artist_scores)

        ith = -1

        nth_artist = scores_argsort[ith]

        while (nth_artist in seen):
            ith = ith-1
            nth_artist = scores_argsort[ith]

        x_cand = X.iloc[nth_artist]
        reward = target[nth_artist]
        x = x_cand
        # Bayesian Update rule
        A = A + x @ x.T

        b = b + reward * x

        seen.append(nth_artist)
        decision_sequence.append((X.index[nth_artist], reward))

        if reward == 1:
            accumulated_rewards += 1

        if target[random_artist] == 1:
            control_rewards += 1

    experiment_data[uid]["decision_sequence"] = decision_sequence
    experiment_data[uid]["control_rewards"] = control_rewards
    experiment_data[uid]["accumulated_rewards"] = accumulated_rewards
    experiment_data[uid][f"theta_t_{H}"] = list(np.round(theta, 3))

    print(
        f"User: {uid} | Rewards: {accumulated_rewards} | Control: {control_rewards}")

with open(f"linearucb_experiment_data_{rounds}.json", "w") as f:
    f.write(json.dumps(experiment_data,  cls=NpEncoder))


component_interpretation_matrix = pd.DataFrame(
    VT, columns=original_Av_columns).T

component_interpretation_matrix.to_csv("component_interpretation_matrix.csv")
