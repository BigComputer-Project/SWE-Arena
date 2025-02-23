from log_reader import RemoteLogReader
from fastchat.serve.model_sampling import SAMPLING_WEIGHTS

from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import os


def get_model_count(output_path='./output'):
    start_date = datetime.strptime("2025_02_03", "%Y_%m_%d")
    end_date = datetime.today()
    log_reader = RemoteLogReader()
    model_list = list(SAMPLING_WEIGHTS.keys())
    current_date = start_date
    model_count = defaultdict(int)
    battle_count = defaultdict(lambda: defaultdict(int))

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
        current_date += timedelta(days=1)

    os.makedirs(output_path, exist_ok=True)
    model_count = {m: model_count[m] for m in model_list}
    battle_count = {m1: {m2: battle_count[m1][m2] + battle_count[m2][m1] for m2 in model_list} for m1 in model_list}
    json.dump(model_count, open(output_path + '/model_count.json', 'w'), indent=4)
    json.dump(battle_count, open(output_path + '/battle_count.json', 'w'), indent=4)


if __name__ == '__main__':
    get_model_count()
