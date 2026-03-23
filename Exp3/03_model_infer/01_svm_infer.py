#%%
from sklearn import svm
import pandas as pd
import numpy as np
import csv
from pathlib import Path
#%%
def read_data(data_file):
    data = pd.read_csv(data_file)
    data.columns = data.columns.str.strip()
    if 'Emotion' in data.columns:
        data['Emotion'] = data['Emotion'].astype(str).str.strip()
    X = data[['Suddenness', 'Goal_relevance', 'Conduciveness', 'Power']].values
    y = data['Emotion'].values
    return X, y

BASE_DIR = Path(__file__).resolve().parent.parent
X_training, y_training = read_data(BASE_DIR / 'data' / 'classifier_train.csv')
X_testing, y_testing = read_data(BASE_DIR / 'data' / 'model_result.csv')

target_names = list(dict.fromkeys(y_training))

def generate_prediction_result (sample, filename):
# Write header to the output file
    with open(filename, 'w') as new_file:
        writer = csv.writer(new_file)
        fieldnames = ['C', 'Story', 'Emotion', 'Val']
        writer.writerow(fieldnames)

    for c in sample:
        svc = svm.SVC(kernel='linear', C=c, probability=True).fit(X_training,y_training)
        for i in range(4):
            for e, target_name in enumerate(target_names):
                with open(filename, 'a', newline='') as new_file:
                    writer = csv.writer(new_file)
                    prob = svc.predict_proba(X_testing[i].reshape(1, -1))[0][e]
                    writer.writerow([c, target_names[i], target_name, prob])

# Generate random samples from a normal distribution for the C parameter
# A normal distribution for c with mean=0.0013, var = 0.0001
# 34 represents modeling 34 participants

# Define output filename
filename_free = BASE_DIR / 'data' / 'svm_free_0.0013_var.csv'
samples = np.random.normal(13, np.sqrt(1), 34) / 10000
generate_prediction_result(samples,filename_free)

# A normal distribution for c with mean =0.0034, var = 0.001
filename_limit = BASE_DIR / 'data' / 'svm_limit_0.0034_var.csv'
samples_limit = np.random.normal(34, np.sqrt(10), 34) / 10000
generate_prediction_result(samples_limit,filename_limit)
    
# %%
# https://scikit-learn.org/stable/modules/svm.html
# https://people.revoledu.com/kardi/tutorial/Python/SVM+in+Python.html
