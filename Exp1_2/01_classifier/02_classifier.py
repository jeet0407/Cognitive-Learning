import csv
from scipy.stats import halfnorm
import numpy as np
from pathlib import Path
import os

def get_seed():
    seed_env = os.getenv("RUN_SEED")
    return int(seed_env) if seed_env else 0

def use_effort():
    return os.getenv("APPRAISAL_VARIANT", "baseline").lower() == "effort"

# Reproducible synthetic sampling for benchmark run.
np.random.seed(get_seed() + 101)

# Random Value Generators
generators = {
    'very_low': lambda: round(halfnorm.rvs(loc=0, scale=0.05), 5),
    'obstruct': lambda: round(halfnorm.rvs(loc=0, scale=0.05), 5),
    'low': lambda: round(halfnorm.rvs(loc=0, scale=0.1), 5),
    'medium': lambda: round(np.random.normal(loc=0.5, scale=0.05), 5),
    'high': lambda: round(1 - halfnorm.rvs(loc=0, scale=0.1), 5),
    'very_high': lambda: round(1 - halfnorm.rvs(loc=0, scale=0.05), 5),
    'open': lambda: round(np.random.uniform(0, 1), 5)
}

def get_random_values(name_list):
    # Generate random values based on the name list
    return [generators[name]() for name in name_list]

def generate_sample_data(n=None, filename=None):
    # Effort prior is only included for effort variant.
    if use_effort():
        emotions = {
            'Boredom': ['very_low', 'low', 'open', 'medium', 'medium'],
            'Fear': ['high', 'high', 'obstruct', 'very_low', 'high'],
            'Happiness': ['low', 'medium', 'high', 'open', 'low'],
            'Joy': ['high', 'high', 'very_high', 'open', 'low'],
            'Pride': ['open', 'high', 'high', 'open', 'medium'],
            'Sadness': ['low', 'high', 'obstruct', 'very_low', 'high'],
            'Shame': ['open', 'high', 'open', 'open', 'high']
        }
        fieldnames = ['Emotion','Suddenness','Goal_relevance','Conduciveness','Power','Effort']
    else:
        emotions = {
            'Boredom': ['very_low', 'low', 'open', 'medium'],
            'Fear': ['high', 'high', 'obstruct', 'very_low'],
            'Happiness': ['low', 'medium', 'high', 'open'],
            'Joy': ['high', 'high', 'very_high', 'open'],
            'Pride': ['open', 'high', 'high', 'open'],
            'Sadness': ['low', 'high', 'obstruct', 'very_low'],
            'Shame': ['open', 'high', 'open', 'open']
        }
        fieldnames = ['Emotion','Suddenness','Goal_relevance','Conduciveness','Power']

    with open(filename, 'w', newline='') as new_file:
        thewriter = csv.writer(new_file)
        thewriter.writerow(fieldnames)
        for emotion, values in emotions.items():
            for i in range(n):
                row = [emotion] + get_random_values(values)
                thewriter.writerow(row)

BASE_DIR = Path(__file__).resolve().parent.parent

filename_train = BASE_DIR / 'data' / 'classifier_train.csv'
generate_sample_data(n=400, filename=filename_train)

filename_test = BASE_DIR / 'data' / 'classifier_test.csv'
generate_sample_data(n=10, filename=filename_test)
