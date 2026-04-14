#%%
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import svm

SEED = 7201
np.random.seed(SEED)


def read_data(data_file):
    data = pd.read_csv(data_file)
    data.columns = data.columns.str.strip()
    if 'Emotion' in data.columns:
        data['Emotion'] = data['Emotion'].astype(str).str.strip()
    x = data[['Suddenness', 'Goal_relevance', 'Conduciveness', 'Power', 'Responsibility']].values
    y = data['Emotion'].values
    return x, y


def load_tuned_c(base_dir):
    defaults = {
        "free": {"c_mean": 0.0013, "c_var": 0.0001},
        "limit": {"c_mean": 0.0034, "c_var": 0.001},
    }
    c_file = base_dir / 'data' / 'c_tuned.json'
    if not c_file.exists():
        return defaults
    with open(c_file, 'r', encoding='utf-8') as fp:
        tuned = json.load(fp)
    for key, val in defaults.items():
        if key not in tuned:
            tuned[key] = val
    return tuned


def generate_prediction_result(sample, filename, x_train, y_train, x_test, target_names):
    with open(filename, 'w', newline='') as new_file:
        writer = csv.writer(new_file)
        writer.writerow(['C', 'Story', 'Emotion', 'Val'])

    for c in sample:
        svc = svm.SVC(kernel='linear', C=float(c), probability=True, random_state=SEED).fit(x_train, y_train)
        for i in range(4):
            probs = svc.predict_proba(x_test[i].reshape(1, -1))[0]
            with open(filename, 'a', newline='') as new_file:
                writer = csv.writer(new_file)
                for e, target_name in enumerate(target_names):
                    writer.writerow([float(c), target_names[i], target_name, float(probs[e])])


BASE_DIR = Path(__file__).resolve().parent.parent
X_training, y_training = read_data(BASE_DIR / 'data' / 'classifier_train.csv')
X_testing, y_testing = read_data(BASE_DIR / 'data' / 'model_result.csv')
target_names = list(dict.fromkeys(y_training))
tuned = load_tuned_c(BASE_DIR)

filename_free = BASE_DIR / 'data' / 'svm_free_0.0013_var.csv'
samples = np.random.normal(
    tuned['free']['c_mean'],
    np.sqrt(max(tuned['free']['c_var'], 1e-8)),
    34
)
generate_prediction_result(samples, filename_free, X_training, y_training, X_testing, target_names)

filename_limit = BASE_DIR / 'data' / 'svm_limit_0.0034_var.csv'
samples_limit = np.random.normal(
    tuned['limit']['c_mean'],
    np.sqrt(max(tuned['limit']['c_var'], 1e-8)),
    34
)
generate_prediction_result(samples_limit, filename_limit, X_training, y_training, X_testing, target_names)

#%%
