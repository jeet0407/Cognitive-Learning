#%%
from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

SEED = 9705
np.random.seed(SEED)

#%%
def read_data(data_file):
    data = pd.read_csv(data_file)
    X = data[['Suddenness', 'Goal_relevance', 'Conduciveness', 'Power', 'Experience']].values
    y = data['Emotion'].values
    return X, y

def predict_with_svm(c, X_train, y_train, X_test):
    svc = svm.SVC(kernel='linear', C=c, probability=True, random_state=SEED).fit(X_train, y_train)
    predictions = [svc.predict_proba(x.reshape(1, -1)).tolist()[0] for x in X_test]
    return predictions

def evaluate_c_precision(min_value, max_value, X_train, y_train, X_test, y_test):
    c_values = np.linspace(min_value, max_value, 200)
    avg_correct_prediction_rates = []
    target_names = list(dict.fromkeys(y_train))

    for c in c_values:
        predictions = predict_with_svm(c, X_train, y_train, X_test)
        correct_prediction_rates = []
        for i, prediction in enumerate(predictions):
            target_index = target_names.index(y_test[i])
            correct_prediction_rates.append(prediction[target_index])
        avg_correct_prediction_rates.append(np.mean(correct_prediction_rates))

    return c_values, np.array(avg_correct_prediction_rates)

def recommend_c(c_values, rates, target_mean):
    idx = int(np.argmin(np.abs(rates - target_mean)))
    c_mean = float(c_values[idx])
    neighborhood = c_values[max(0, idx - 5):min(len(c_values), idx + 6)]
    c_var = float(max(np.var(neighborhood), 1e-8))
    return c_mean, c_var

BASE_DIR = Path(__file__).resolve().parent.parent
X_training, y_training = read_data(BASE_DIR / 'data' / 'classifier_train.csv')
X_testing, y_testing = read_data(BASE_DIR / 'data' / 'classifier_test.csv')
c_values, rates = evaluate_c_precision(0.001, 0.03, X_training, y_training, X_testing, y_testing)

plt.plot(c_values, rates)
plt.xlabel('C value')
plt.ylabel('Average correct prediction rate')
plt.tight_layout()
plt.savefig(BASE_DIR / 'plots' / 'classifier_c_curve.png', dpi=200)

# Human precision anchors from original paper setup.
targets = {
    "free": 0.4019302024931973,
    "limit": 0.8714285714285714,
}

result = {}
for key, target in targets.items():
    c_mean, c_var = recommend_c(c_values, rates, target)
    result[key] = {"c_mean": c_mean, "c_var": c_var, "target_mean": target}
    print(f"Recommended {key} c_mean={c_mean:.6f}, c_var={c_var:.8f}")

with open(BASE_DIR / 'data' / 'c_tuned.json', 'w', encoding='utf-8') as fp:
    json.dump(result, fp, indent=2)

#%%
