from log_reader import RemoteLogReader
from fastchat.serve.model_sampling import SAMPLING_WEIGHTS

from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from tqdm import tqdm
import random


def get_model_count(output_path='./output'):
    start_date = datetime.strptime("2025_02_03", "%Y_%m_%d")
    end_date = datetime.today()
    log_reader = RemoteLogReader()
    model_list = list(SAMPLING_WEIGHTS.keys())
    current_date = start_date
    model_count = defaultdict(int)
    battle_count = defaultdict(lambda: defaultdict(int))
    matches = []

    while current_date <= end_date:
        logs = log_reader.get_conv_logs(current_date.strftime("%Y_%m_%d"))
        s_logs = log_reader.get_sandbox_logs(current_date.strftime("%Y_%m_%d"))
        vali_conv_ids = [log['sandbox_state']['conv_id'] for log in s_logs]
        for task in ('battle_anony', 'battle_named'):
            for log in logs[task].values():
                for record in log:
                    if 'vote' in record['type'] and record['states'][0]['conv_id'] in vali_conv_ids and record['states'][1]['conv_id'] in vali_conv_ids:
                        model1 = record['states'][0]['model_name']
                        model2 = record['states'][1]['model_name']
                        if model1 in model_list and model2 in model_list:
                            model_count[model1] += 1
                            model_count[model2] += 1
                            battle_count[model1][model2] += 1
                            matches.append([model1, model2, record['feedback']['vote_type']])
        current_date += timedelta(days=1)

    os.makedirs(output_path, exist_ok=True)
    model_count = {m: model_count[m] for m in model_list}
    battle_count = {m1: {m2: battle_count[m1][m2] + battle_count[m2][m1] for m2 in model_list} for m1 in model_list}
    json.dump(model_count, open(output_path + '/model_count.json', 'w'), indent=4)
    json.dump(battle_count, open(output_path + '/battle_count.json', 'w'), indent=4)
    json.dump(matches, open(output_path + '/matches.json', 'w'), indent=4)


def compute_online_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    rating = defaultdict(lambda: INIT_RATING)
    for model_a, model_b, vote in battles:
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if vote == "vote_left":
            sa = 1
        elif vote == "vote_right":
            sa = 0
        else:
            sa = 0.5
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)
    return dict(rating)


def bootstrap_confidence_intervals(output_path='./output', n_bootstrap=10000, confidence_level=0.9, K=4, SCALE=400,
                                   BASE=10, INIT_RATING=1000):
    battles = json.load(open(output_path + '/matches.json', 'r'))
    original_ratings = compute_online_elo(battles, K, SCALE, BASE, INIT_RATING)
    bootstrap_ratings = defaultdict(list)

    for _ in tqdm(range(n_bootstrap)):
        bootstrap_sample = random.choices(battles, k=len(battles))
        sample_ratings = compute_online_elo(bootstrap_sample, K, SCALE, BASE, INIT_RATING)
        for model, rating in sample_ratings.items():
            bootstrap_ratings[model].append(rating)

    alpha = (1 - confidence_level) / 2
    result = {}

    for model, ratings in bootstrap_ratings.items():
        ratings_sorted = sorted(ratings)
        lower_idx = int(alpha * n_bootstrap)
        upper_idx = int((1 - alpha) * n_bootstrap)

        result[model] = {
            'rating': original_ratings[model],
            'lower_bound': ratings_sorted[lower_idx],
            'upper_bound': ratings_sorted[upper_idx],
            'ci': [ratings_sorted[lower_idx] - original_ratings[model], ratings_sorted[upper_idx] - original_ratings[model]]
        }

    sorted_result = dict(sorted(result.items(), key=lambda x: x[1]['rating'], reverse=True))
    json.dump(sorted_result, open(output_path + '/elo.json', 'w'), indent=4)


if __name__ == '__main__':
    bootstrap_confidence_intervals()
