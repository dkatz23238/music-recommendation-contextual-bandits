import glob
import pandas as pd
from typing import Tuple


def read_dfs(
    glob_path="./hetrec2011-lastfm-2k/*.dat",
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    paths = glob.glob(glob_path)

    dframes = {}

    for p in paths:
        df = pd.read_csv(p, delimiter="\t", on_bad_lines="skip", encoding="latin-1")
        dframes[p.split("/")[-1].replace(".dat", "")] = df

    dframes["artists"]["id"] = dframes["artists"]["id"].astype(int)

    # User Artist Pairs
    uap = pd.merge(
        dframes["user_artists"], dframes["artists"], left_on="artistID", right_on="id"
    )

    # User tag pairs
    user_tag_pairs = pd.merge(
        dframes["user_taggedartists"],
        dframes["tags"],
        left_on="tagID",
        right_on="tagID",
    )

    # Artist tag pair

    atp = pd.merge(
        user_tag_pairs, dframes["artists"], left_on="artistID", right_on="id"
    )

    uf = dframes["user_friends"]

    return atp, uap, uf
