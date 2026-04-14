#%%
import numpy as np
import csv
from scipy.stats import halfnorm
from pathlib import Path
#%%

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

urgency_generators = {
    'low_urgency': lambda: round(np.random.uniform(0.0, 0.3), 5),
    'medium_urgency': lambda: round(np.random.uniform(0.4, 0.6), 5),
    'high_urgency': lambda: round(np.random.uniform(0.8, 1.0), 5),
}

effort_generators = {
    'low_effort': lambda: round(np.random.uniform(0.0, 0.3), 5),
    'medium_effort': lambda: round(np.random.uniform(0.4, 0.6), 5),
    'high_effort': lambda: round(np.random.uniform(0.7, 1.0), 5),
}

def get_random_values(name_list):
    # Generate random values based on the name list
    return [generators[name]() for name in name_list]

def get_urgency_value(urgency_name):
    return urgency_generators[urgency_name]()

def get_effort_value(effort_name):
    return effort_generators[effort_name]()

def generate_sample_data(n=None, filename=None):
    emotions = {
        'Anxiety':[ 'low', 'medium', 'obstruct', 'low', 'high_urgency', 'medium_effort'],
        'Despair':[ 'high', 'high', 'obstruct', 'very_low', 'high_urgency', 'medium_effort'],
        'Irritation': ['low', 'medium', 'obstruct', 'medium', 'medium_urgency', 'medium_effort'],
        'Rage': ['high', 'high', 'obstruct', 'high', 'high_urgency', 'high_effort']
    }

    with open(filename, 'w', newline='') as new_file:
        thewriter = csv.writer(new_file)
        fieldnames = ['Emotion','Suddenness','Goal_relevance','Conduciveness','Power','Urgency','Effort']
        thewriter.writerow(fieldnames)
        for emotion, values in emotions.items():
            for i in range(n):
                row = [emotion] + get_random_values(values[:4]) + [get_urgency_value(values[4]), get_effort_value(values[5])]
                thewriter.writerow(row)

BASE_DIR = Path(__file__).resolve().parent.parent

filename_train = BASE_DIR / 'data' / 'classifier_train.csv'
generate_sample_data(n=400, filename=filename_train)

filename_test = BASE_DIR / 'data' / 'classifier_test.csv'
generate_sample_data(n=10, filename=filename_test)

