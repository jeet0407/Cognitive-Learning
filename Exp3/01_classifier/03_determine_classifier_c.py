#%%
from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# we have 70 rows of testing data. 7 emotion, 10 model result for each emotion. 
def read_data (data_file):
    data = pd.read_csv(data_file)
    X = data[feature_columns()].values
    y = data['Emotion'].values
    return X,y

def predict_with_svm(c,x_train, y_train, X_test):
    svc = svm.SVC(kernel='linear', C=c, probability=True, random_state=get_seed() + 401).fit(x_train,y_train)
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
    neighborhood = c_values[max(0, idx-5):min(len(c_values), idx+6)]
    c_var = float(max(np.var(neighborhood), 1e-8))
    return c_mean, c_var

BASE_DIR = Path(__file__).resolve().parent.parent
x_training, y_training = read_data(BASE_DIR / 'data' / 'classifier_train.csv')
x_testing, y_testing = read_data(BASE_DIR / 'data' / 'classifier_test.csv')
c_values, rates = evaluate_c_precision(0.0005,0.008,x_training,y_training,x_testing,y_testing)
plt.plot(c_values, rates)
plt.xlabel('C value')
plt.ylabel('Average correct prediction rate')
plt.show()
#%%
# This is for plotting the classifier with a range of c, so that we can compare the y-axie
# and figure out the suitable c value range for fitting human data.
# From the human data (humandata_for_c.py), we get the result that 
# for precision, mean = 0.4019302024931973, var = 0.027696655833540938

# Here is to find c according to precision plot
# x is a range of c value, y is the precision

# for every c, we have 70 precision added from 70 rows, and then get the average precision



# For exp3 free-rating
# mean = 0.29215564961029417 var = 0.0067888217049672
# c=0.0013, var = 0.0001
# For exp3 limit rating
# mean = 0.6617647058823529 var = 0.22383217993079588
# c=0.0034, var = 0.001

targets = {"free": 0.29215564961029417, "limit": 0.6617647058823529}
result = {}
for key, target in targets.items():
    c_mean, c_var = recommend_c(c_values, rates, target)
    result[key] = {"c_mean": c_mean, "c_var": c_var, "target_mean": target}
    print(f"Recommended {key} c_mean={c_mean:.6f}, c_var={c_var:.8f}")

with open(BASE_DIR / 'data' / 'c_tuned.json', 'w', encoding='utf-8') as fp:
    json.dump(result, fp, indent=2)

#%%

# https://scikit-learn.org/stable/modules/svm.html
# https://people.revoledu.com/kardi/tutorial/Python/SVM+in+Python.html
