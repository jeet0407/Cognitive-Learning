import csv
from scipy.stats import halfnorm
import numpy as np
from pathlib import Path

SEED = 9505
np.random.seed(SEED)

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
    # 5th factor prior: Experience (novice/expert-like familiarity signal).
    # Assumptions for Experience by emotion:
    # - Boredom/Pride: higher familiarity with context or routine.
    # - Fear/Sadness/Shame: lower familiarity (novice-like vulnerability).
    # - Happiness/Joy: mid familiarity.
    emotions = {
        'Boredom': ['very_low', 'low', 'open', 'medium', 'high'],
        'Fear': ['high', 'high', 'obstruct', 'very_low', 'low'],
        'Happiness': ['low', 'medium', 'high', 'open', 'medium'],
        'Joy': ['high', 'high', 'very_high', 'open', 'medium'],
        'Pride': ['open', 'high', 'high', 'open', 'very_high'],
        'Sadness': ['low', 'high', 'obstruct', 'very_low', 'low'],
        'Shame': ['open', 'high', 'open', 'open', 'low']
    }

    with open(filename, 'w', newline='') as new_file:
        thewriter = csv.writer(new_file)
        fieldnames = ['Emotion','Suddenness','Goal_relevance','Conduciveness','Power','Experience']
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
