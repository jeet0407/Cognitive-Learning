#%%
import numpy as np
import csv
from scipy.stats import halfnorm
from pathlib import Path
#%%

# Reproducible synthetic sampling for Effort ablation.
SEED = 9603
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
    # Added 5th factor prior: Effort
    # Rage/despair are modeled with higher burden than irritation.
    emotions = {
        'Anxiety':[ 'low', 'medium', 'obstruct', 'low', 'high'],
        'Despair':[ 'high', 'high', 'obstruct', 'very_low', 'very_high'],
        'Irritation': ['low', 'medium', 'obstruct', 'medium', 'medium'],
        'Rage': ['high', 'high', 'obstruct', 'high', 'very_high']
    }

    with open(filename, 'w', newline='') as new_file:
        thewriter = csv.writer(new_file)
        fieldnames = ['Emotion','Suddenness','Goal_relevance','Conduciveness','Power','Effort']
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

