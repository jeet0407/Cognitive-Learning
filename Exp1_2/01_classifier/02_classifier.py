import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common.five_dim_pipeline import generate_gmm_dataset

EMOTION_PROFILES = {
    'Boredom': {
        'weights': [0.7, 0.3],
        'means': [
            [0.08, 0.18, 0.48, 0.52, 0.12],
            [0.13, 0.24, 0.54, 0.44, 0.18],
        ],
        'scales': [
            [0.04, 0.06, 0.05, 0.06, 0.04],
            [0.05, 0.06, 0.05, 0.07, 0.05],
        ],
    },
    'Fear': {
        'weights': [0.65, 0.35],
        'means': [
            [0.88, 0.90, 0.08, 0.10, 0.90],
            [0.78, 0.82, 0.12, 0.06, 0.82],
        ],
        'scales': [
            [0.05, 0.05, 0.04, 0.04, 0.05],
            [0.06, 0.06, 0.04, 0.03, 0.06],
        ],
    },
    'Happiness': {
        'weights': [0.7, 0.3],
        'means': [
            [0.15, 0.55, 0.84, 0.55, 0.12],
            [0.10, 0.48, 0.78, 0.48, 0.18],
        ],
        'scales': [
            [0.05, 0.06, 0.05, 0.06, 0.04],
            [0.05, 0.06, 0.05, 0.06, 0.05],
        ],
    },
    'Joy': {
        'weights': [0.7, 0.3],
        'means': [
            [0.85, 0.88, 0.95, 0.58, 0.10],
            [0.78, 0.80, 0.90, 0.50, 0.16],
        ],
        'scales': [
            [0.05, 0.05, 0.04, 0.06, 0.04],
            [0.05, 0.05, 0.04, 0.06, 0.05],
        ],
    },
    'Pride': {
        'weights': [0.6, 0.4],
        'means': [
            [0.48, 0.82, 0.84, 0.62, 0.52],
            [0.55, 0.90, 0.76, 0.70, 0.58],
        ],
        'scales': [
            [0.06, 0.05, 0.05, 0.06, 0.06],
            [0.06, 0.04, 0.05, 0.06, 0.05],
        ],
    },
    'Sadness': {
        'weights': [0.65, 0.35],
        'means': [
            [0.12, 0.88, 0.08, 0.08, 0.48],
            [0.18, 0.82, 0.12, 0.12, 0.42],
        ],
        'scales': [
            [0.05, 0.05, 0.04, 0.04, 0.05],
            [0.05, 0.06, 0.04, 0.05, 0.05],
        ],
    },
    'Shame': {
        'weights': [0.6, 0.4],
        'means': [
            [0.42, 0.86, 0.38, 0.34, 0.55],
            [0.35, 0.78, 0.48, 0.28, 0.48],
        ],
        'scales': [
            [0.06, 0.05, 0.06, 0.06, 0.05],
            [0.06, 0.06, 0.06, 0.06, 0.05],
        ],
    },
}

BASE_DIR = Path(__file__).resolve().parent.parent

train_data = generate_gmm_dataset(EMOTION_PROFILES, samples_per_emotion=400)
train_data.to_csv(BASE_DIR / 'data' / 'classifier_train.csv', index=False)

test_data = generate_gmm_dataset(EMOTION_PROFILES, samples_per_emotion=10, seed=101)
test_data.to_csv(BASE_DIR / 'data' / 'classifier_test.csv', index=False)
