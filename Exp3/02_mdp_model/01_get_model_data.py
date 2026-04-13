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
    fieldnames = ['Emotion', 'Suddenness', 'Goal_relevance', 'Conduciveness', 
    'Power', 'Urgency' ]
    writer.writerow(fieldnames)

#%%
program_list = [
    BASE_DIR / '02_mdp_model' / 'anxiety.py',
    BASE_DIR / '02_mdp_model' / 'despair.py',
    BASE_DIR / '02_mdp_model' / 'irritation.py',
    BASE_DIR / '02_mdp_model' / 'rage.py',
]

for program in program_list:
    subprocess.call([sys.executable, str(program)])
    print("Finished:" + str(program))
