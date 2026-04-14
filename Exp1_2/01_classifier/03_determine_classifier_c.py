import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common.five_dim_pipeline import read_xy, run_rbf_grid_search, save_tuning, stratified_subsample

FREE_TARGET = {'mean': 0.4019302024931973, 'var': 0.027696655833540938}
LIMIT_TARGET = {'mean': 0.8714285714285714, 'var': 0.11204081632653061}

BASE_DIR = Path(__file__).resolve().parent.parent
X_training, y_training = read_xy(BASE_DIR / 'data' / 'classifier_train.csv')
X_testing, y_testing = read_xy(BASE_DIR / 'data' / 'classifier_test.csv')
X_tune, y_tune = stratified_subsample(X_training, y_training, max_per_class=160)

c_grid = np.logspace(-3, -0.2, 8)
gamma_grid = ['scale', 0.03, 0.1, 0.3]

free_tuning, free_results = run_rbf_grid_search(
    X_tune,
    y_tune,
    X_testing,
    y_testing,
    FREE_TARGET['mean'],
    FREE_TARGET['var'],
    c_grid,
    gamma_grid,
)

limit_tuning, limit_results = run_rbf_grid_search(
    X_tune,
    y_tune,
    X_testing,
    y_testing,
    LIMIT_TARGET['mean'],
    LIMIT_TARGET['var'],
    c_grid,
    gamma_grid,
)

save_tuning(BASE_DIR / 'data' / 'rbf_tuning.json', free_tuning, limit_tuning)

print('Best free-rating candidates:')
print(free_results.head(10).round(6).to_string(index=False))
print('\nDerived free C mean / var:')
print(round(free_tuning['c_mean'], 6), round(free_tuning['c_var'], 8))

print('\nBest limited-rating candidates:')
print(limit_results.head(10).round(6).to_string(index=False))
print('\nDerived limit C mean / var:')
print(round(limit_tuning['c_mean'], 6), round(limit_tuning['c_var'], 8))
