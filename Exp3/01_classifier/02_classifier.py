import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common.five_dim_pipeline import generate_gmm_dataset

EMOTION_PROFILES = {
    'Anxiety': {
        'weights': [0.65, 0.35],
        'means': [
            [0.28, 0.62, 0.10, 0.24, 0.86],
            [0.18, 0.55, 0.15, 0.20, 0.78],
        ],
        'scales': [
            [0.05, 0.06, 0.04, 0.05, 0.05],
            [0.05, 0.06, 0.04, 0.05, 0.06],
        ],
    },
    'Despair': {
        'weights': [0.7, 0.3],
        'means': [
            [0.82, 0.90, 0.06, 0.06, 0.86],
            [0.74, 0.82, 0.10, 0.10, 0.74],
        ],
        'scales': [
            [0.05, 0.05, 0.04, 0.04, 0.05],
            [0.06, 0.05, 0.04, 0.05, 0.06],
        ],
    },
    'Irritation': {
        'weights': [0.6, 0.4],
        'means': [
            [0.26, 0.56, 0.12, 0.48, 0.54],
            [0.18, 0.48, 0.18, 0.56, 0.62],
        ],
        'scales': [
            [0.05, 0.06, 0.05, 0.06, 0.05],
            [0.05, 0.06, 0.05, 0.06, 0.05],
        ],
    },
    'Rage': {
        'weights': [0.7, 0.3],
        'means': [
            [0.86, 0.90, 0.06, 0.82, 0.90],
            [0.76, 0.84, 0.10, 0.72, 0.82],
        ],
        'scales': [
            [0.05, 0.05, 0.04, 0.06, 0.05],
            [0.06, 0.05, 0.04, 0.06, 0.05],
        ],
    },
}

BASE_DIR = Path(__file__).resolve().parent.parent

train_data = generate_gmm_dataset(EMOTION_PROFILES, samples_per_emotion=400)
train_data.to_csv(BASE_DIR / 'data' / 'classifier_train.csv', index=False)

test_data = generate_gmm_dataset(EMOTION_PROFILES, samples_per_emotion=10, seed=101)
test_data.to_csv(BASE_DIR / 'data' / 'classifier_test.csv', index=False)
