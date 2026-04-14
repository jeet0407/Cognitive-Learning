import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common.five_dim_pipeline import (
    RANDOM_STATE,
    load_tuning,
    read_xy,
    sampled_c_values,
    write_probability_table,
)

BASELINE = {
    'free': {'c_mean': 0.0032, 'c_var': 0.0002, 'sample_size': 42},
    'limit': {'c_mean': 0.014, 'c_var': 0.0056, 'sample_size': 30},
}


def build_predictor(X_train, y_train, X_test, kernel, gamma_value, c_value):
    pipeline = Pipeline(
        [
            ('scaler', MinMaxScaler()),
            (
                'svc',
                SVC(
                    kernel=kernel,
                    C=float(c_value),
                    gamma=gamma_value,
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline.predict_proba(X_test)


BASE_DIR = Path(__file__).resolve().parent.parent
X_training, y_training = read_xy(BASE_DIR / 'data' / 'classifier_train.csv')
X_testing, y_testing = read_xy(BASE_DIR / 'data' / 'model_result.csv')
target_names = list(dict.fromkeys(y_training))
tuning = load_tuning(BASE_DIR / 'data' / 'rbf_tuning.json')


write_probability_table(
    BASE_DIR / 'data' / 'svm_free_linear_baseline.csv',
    y_testing,
    target_names,
    sampled_c_values(BASELINE['free']['c_mean'], BASELINE['free']['c_var'], BASELINE['free']['sample_size'], seed_offset=1),
    lambda c_value: build_predictor(X_training, y_training, X_testing, 'linear', 'scale', c_value),
)

write_probability_table(
    BASE_DIR / 'data' / 'svm_limit_linear_baseline.csv',
    y_testing,
    target_names,
    sampled_c_values(BASELINE['limit']['c_mean'], BASELINE['limit']['c_var'], BASELINE['limit']['sample_size'], seed_offset=2),
    lambda c_value: build_predictor(X_training, y_training, X_testing, 'linear', 'scale', c_value),
)

write_probability_table(
    BASE_DIR / 'data' / 'svm_free_rbf_opt.csv',
    y_testing,
    target_names,
    sampled_c_values(tuning['free']['c_mean'], tuning['free']['c_var'], BASELINE['free']['sample_size'], seed_offset=3),
    lambda c_value: build_predictor(X_training, y_training, X_testing, 'rbf', tuning['free']['best_gamma'], c_value),
)

write_probability_table(
    BASE_DIR / 'data' / 'svm_limit_rbf_opt.csv',
    y_testing,
    target_names,
    sampled_c_values(tuning['limit']['c_mean'], tuning['limit']['c_var'], BASELINE['limit']['sample_size'], seed_offset=4),
    lambda c_value: build_predictor(X_training, y_training, X_testing, 'rbf', tuning['limit']['best_gamma'], c_value),
)
