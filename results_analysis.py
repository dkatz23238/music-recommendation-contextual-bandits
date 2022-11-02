import json
import numpy as np
import glob

runs = []
meta_keys = ["use_wiki_data", "use_only_wiki_data"]

for f in glob.glob("linearucb_experiment_data_*.json"):
    with open(f) as fhandle:

        run = json.loads(fhandle.read())
        control = [
            run[k]["control_rewards"] for k in run if k not in meta_keys
        ]
        reward = [
            run[k]["accumulated_rewards"] for k in run if k not in meta_keys
        ]

        for mk in meta_keys:
            print(f"{mk} {run[mk]}")

        r = round(sum(reward) / (35*150), 2)
        c = round(sum(control) / (35*150), 2)

        lift = r-c
        print(
            f"control: {c} | reward: {r} | lift: {lift}\n"
        )
        runs.append(run)
