import random
from readers import read_dfs
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

from time import perf_counter

TOP_TAGS = int(sys.argv[1])
TOP_ARTIST_POOL = int(sys.argv[2])
T_ROUNDS = int(sys.argv[3])
ALPHA = float(sys.argv[4])
N_TRIALS = int(sys.argv[5])

atp, uap, uf = read_dfs()

tags_counts = (atp.groupby("tagID")
               .count()[["artistID"]]
               .sort_values("artistID"))

admissable_tag_ids = tags_counts.iloc[-TOP_TAGS:, :].index

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

lifts = []

trials = []

for user_id in np.random.randint(1, 2100, size=N_TRIALS):
    t1_start = perf_counter()
    user_artists = uap.query("userID==@user_id").name.values
    artists = popular_artists.index.values[-TOP_ARTIST_POOL:]

    x_list = []
    target = []

    for a in artists:
        if a in artist_vectors.index:
            x_list.append(artist_vectors.loc[a])
            if a in user_artists:
                target.append(1)
            else:
                target.append(-1)

    X = pd.concat(x_list, axis=1).T
    target = np.array(target)

    alpha = ALPHA

    A = np.identity(X.shape[1])
    b = np.zeros(X.shape[1])

    seen = []

    cumm_rewards = 0
    control_rewards = 0

    decision_sequence = []

    for t in range(T_ROUNDS):

        theta = np.linalg.inv(A) @ b

        p_ta = theta

        mus = X.values.dot(theta)

        std = alpha * np.sqrt(
            np.diag(X.dot(np.linalg.inv(A)).values @ X.T)
        )

        artist_scores = mus + std  # Upper confidence bound of reward

        random_artist = random.choice(X.index)

        ith = -1
        nth_artist = np.argsort(artist_scores)[ith]

        while (nth_artist in seen):
            ith = ith-1
            nth_artist = np.argsort(artist_scores)[ith]

        x_cand = X.iloc[nth_artist]
        reward = target[nth_artist]

        decision_sequence.append((X.index[nth_artist], reward))

        # Bayesian Update rule
        A = A + x_cand @ x_cand.T
        # A = A + x_cand.values[:, None].dot(x_cand.values[:, None].T)
        b = b + reward * x_cand

        seen.append(nth_artist)

        if reward == 1:
            cumm_rewards += 1

        if random_artist in user_artists:
            control_rewards += 1

    lift = (cumm_rewards - control_rewards)/T_ROUNDS
    lifts.append(lift)
    t1_end = perf_counter()
    trials.append(decision_sequence)
    # print(user_id, lift, round(t1_end-t1_start, 2))

print(np.mean(lifts))
