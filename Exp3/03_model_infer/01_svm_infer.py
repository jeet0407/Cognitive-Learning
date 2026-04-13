#%%
from sklearn import svm
import pandas as pd
import numpy as np
import csv
from pathlib import Path
import json
import os
#%%
def get_seed():
    seed_env = os.getenv("RUN_SEED")
    return int(seed_env) if seed_env else 0

def feature_columns():
    cols = ['Suddenness', 'Goal_relevance', 'Conduciveness', 'Power']
    if os.getenv("APPRAISAL_VARIANT", "baseline").lower() == "effort":
        cols.append('Effort')
    return cols

np.random.seed(get_seed() + 601)

def read_data(data_file):
    data = pd.read_csv(data_file)
    data.columns = data.columns.str.strip()
    if 'Emotion' in data.columns:
        data['Emotion'] = data['Emotion'].astype(str).str.strip()
    X = data[feature_columns()].values
    y = data['Emotion'].values
    return X, y

BASE_DIR = Path(__file__).resolve().parent.parent
X_training, y_training = read_data(BASE_DIR / 'data' / 'classifier_train.csv')
X_testing, y_testing = read_data(BASE_DIR / 'data' / 'model_result.csv')

target_names = list(dict.fromkeys(y_training))

def load_tuned_c():
    defaults = {
        "free": {"c_mean": 0.0013, "c_var": 0.0001},
        "limit": {"c_mean": 0.0034, "c_var": 0.001},
    }
    c_file = BASE_DIR / 'data' / 'c_tuned.json'
    if not c_file.exists():
        return defaults
    with open(c_file, 'r', encoding='utf-8') as fp:
        tuned = json.load(fp)
    for k, v in defaults.items():
        if k not in tuned:
            tuned[k] = v
    return tuned

def generate_prediction_result (sample, filename):
# Write header to the output file
    with open(filename, 'w') as new_file:
        writer = csv.writer(new_file)
        fieldnames = ['C', 'Story', 'Emotion', 'Val']
        writer.writerow(fieldnames)

    for c in sample:
        svc = svm.SVC(
            kernel='linear', C=c, probability=True, random_state=get_seed() + 602
        ).fit(X_training,y_training)
        for i in range(4):
            for e, target_name in enumerate(target_names):
                with open(filename, 'a', newline='') as new_file:
                    writer = csv.writer(new_file)
                    prob = svc.predict_proba(X_testing[i].reshape(1, -1))[0][e]
                    writer.writerow([c, target_names[i], target_name, prob])

# Generate random samples from a normal distribution for the C parameter
# Define output filename
filename_free = BASE_DIR / 'data' / 'svm_free_0.0013_var.csv'
tuned = load_tuned_c()
samples = np.random.normal(
    tuned['free']['c_mean'],
    np.sqrt(max(tuned['free']['c_var'], 1e-8)),
    34
)
generate_prediction_result(samples,filename_free)

filename_limit = BASE_DIR / 'data' / 'svm_limit_0.0034_var.csv'
samples_limit = np.random.normal(
    tuned['limit']['c_mean'],
    np.sqrt(max(tuned['limit']['c_var'], 1e-8)),
    34
)
generate_prediction_result(samples_limit,filename_limit)
    
# %%
# https://scikit-learn.org/stable/modules/svm.html
# https://people.revoledu.com/kardi/tutorial/Python/SVM+in+Python.html
