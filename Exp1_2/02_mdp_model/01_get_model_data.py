#%%
import subprocess
import csv
from pathlib import Path
import sys

#%%

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_RESULT_FILE = BASE_DIR / 'data' / 'model_result.csv'

with open(MODEL_RESULT_FILE, "w") as new_file:
    writer = csv.writer(new_file)
    fieldnames = ['Emotion', 'Suddenness', 'Goal_relevance', 'Conduciveness', 'Power', 'Urgency', 'Effort' ]
    writer.writerow(fieldnames)

#%%
program_list = [
    BASE_DIR / '02_mdp_model' / 'mdp_boredom.py',
    BASE_DIR / '02_mdp_model' / 'mdp_fear.py',
    BASE_DIR / '02_mdp_model' / 'mdp_happiness.py',
    BASE_DIR / '02_mdp_model' / 'mdp_joy.py',
    BASE_DIR / '02_mdp_model' / 'mdp_pride.py',
    BASE_DIR / '02_mdp_model' / 'mdp_sadness.py',
    BASE_DIR / '02_mdp_model' / 'mdp_shame.py',
]

for program in program_list:
    subprocess.call([sys.executable, str(program)])
    print("Finished:" + str(program))
