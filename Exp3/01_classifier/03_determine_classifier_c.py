#%%
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm

SEED = 6201
np.random.seed(SEED)


def read_data(data_file):
    data = pd.read_csv(data_file)
    x = data[['Suddenness', 'Goal_relevance', 'Conduciveness', 'Power', 'Responsibility']].values
    y = data['Emotion'].values
    return x, y


def predict_with_svm(c, x_train, y_train, x_test):
    svc = svm.SVC(kernel='linear', C=c, probability=True, random_state=SEED).fit(x_train, y_train)
    return [svc.predict_proba(x.reshape(1, -1)).tolist()[0] for x in x_test]


def evaluate_c_precision(min_value, max_value, x_train, y_train, x_test, y_test):
    c_values = np.linspace(min_value, max_value, 200)
    avg_correct = []
    target_names = list(dict.fromkeys(y_train))
    for c in c_values:
        predictions = predict_with_svm(c, x_train, y_train, x_test)
        corr = []
        for i, prediction in enumerate(predictions):
            idx = target_names.index(y_test[i])
            corr.append(prediction[idx])
        avg_correct.append(np.mean(corr))
    return c_values, np.array(avg_correct)


def recommend_c(c_values, rates, target_mean):
    idx = int(np.argmin(np.abs(rates - target_mean)))
    c_mean = float(c_values[idx])
    neighborhood = c_values[max(0, idx - 5):min(len(c_values), idx + 6)]
    c_var = float(max(np.var(neighborhood), 1e-8))
    return c_mean, c_var


BASE_DIR = Path(__file__).resolve().parent.parent
x_train, y_train = read_data(BASE_DIR / 'data' / 'classifier_train.csv')
x_test, y_test = read_data(BASE_DIR / 'data' / 'classifier_test.csv')
c_values, rates = evaluate_c_precision(0.0005, 0.008, x_train, y_train, x_test, y_test)

plt.plot(c_values, rates)
plt.xlabel('C value')
plt.ylabel('Average correct prediction rate')
plt.tight_layout()
plt.savefig(BASE_DIR / 'plots' / 'classifier_c_curve.png', dpi=200)

# Human precision anchors from project setup
targets = {"free": 0.29215564961029417, "limit": 0.6617647058823529}
result = {}
for key, target in targets.items():
    c_mean, c_var = recommend_c(c_values, rates, target)
    result[key] = {"c_mean": c_mean, "c_var": c_var, "target_mean": target}
    print(f"Recommended {key} c_mean={c_mean:.6f}, c_var={c_var:.8f}")

with open(BASE_DIR / 'data' / 'c_tuned.json', 'w', encoding='utf-8') as fp:
    json.dump(result, fp, indent=2)

#%%
