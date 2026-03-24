# Cognitive-Learning (Appraisal-RL)

This repository implements an appraisal-based reinforcement learning pipeline and compares model-generated emotion patterns with human ratings.

It contains two experiment folders:

- `Exp1_2`: 7-emotion setting (`Happiness, Joy, Pride, Boredom, Fear, Sadness, Shame`)
- `Exp3`: 4-emotion setting (`Anxiety, Despair, Irritation, Rage`)

---

## 1) End-to-end workflow (both experiments)

Each experiment follows the same 4-stage pipeline:

1. **Classifier setup** (`01_classifier`)
	 - analyze human ratings
	 - generate synthetic classifier train/test data
	 - inspect suitable SVM `C` range
2. **MDP model execution** (`02_mdp_model`)
	 - run emotion-specific RL scenarios
	 - output appraisal vectors to `data/model_result.csv`
3. **Model emotion inference** (`03_model_infer`)
	 - train linear SVM on synthetic classifier data
	 - infer emotion probabilities for model appraisal vectors
4. **Statistical analysis + plots** (`04_statistical_analysis`)
	 - compare human vs model distributions
	 - mixed-effects modeling, CI bars, RMSE/scatter diagnostics

---

## 2) Quick run order

Run inside each experiment folder (e.g., `Exp1_2` or `Exp3`):

1. `01_classifier/01_analyze_human_data.py`
2. `01_classifier/02_classifier.py`
3. `01_classifier/03_determine_classifier_c.py`
4. `02_mdp_model/01_get_model_data.py`
5. `03_model_infer/01_svm_infer.py`
6. `04_statistical_analysis/<experiment analysis>.py` (or `.R`)

---

## 3) Project structure and file responsibilities

## Root

- `README.md` (this file): full project-level documentation.

---

## `Exp1_2` (7 emotions)

### `Exp1_2/readme.md`

- Experiment-specific run notes and expected outputs.

### `Exp1_2/01_classifier`

- `01_analyze_human_data.py`
	- Reads `data/human_free.csv` and `data/human_limit.csv`.
	- Extracts per-sample “correct emotion” precision (diagonal by story label).
	- Prints mean and variance for free and limited rating conditions.

- `02_classifier.py`
	- Generates synthetic appraisal samples for classifier training/testing.
	- Uses distribution templates (`very_low`, `low`, `medium`, `high`, `very_high`, `open`, `obstruct`).
	- Writes:
		- `data/classifier_train.csv`
		- `data/classifier_test.csv`

- `03_determine_classifier_c.py`
	- Trains linear SVM across a sweep of `C` values.
	- Computes average correct-class probability on test samples.
	- Plots `C` vs precision trend to select human-like `C` ranges.

### `Exp1_2/02_mdp_model`

- `agent.py`
	- Shared RL agent for all Exp1_2 scenarios.
	- Implements Q-learning (`alpha`, `gamma`), epsilon-greedy choice, TD error tracking.
	- Computes appraisal dimensions from agent dynamics:
		- `Suddenness`
		- `Goal_relevance`
		- `Conduciveness`
		- `Power`

- `01_get_model_data.py`
	- Initializes `data/model_result.csv` header.
	- Sequentially runs each emotion-specific scenario script.

- `mdp_happiness.py`
	- Happiness MDP transitions/rewards; story-mode intervention.
	- Appends one appraisal row for `Happiness`.

- `mdp_joy.py`
	- Joy scenario with stochastic branch then story forcing.
	- Appends row for `Joy`.

- `mdp_pride.py`
	- Pride scenario with enhanced-goal branch (`G_plus`).
	- Appends row for `Pride`.

- `mdp_boredom.py`
	- Boredom scenario with choice-point behavior.
	- Appends row for `Boredom`.

- `mdp_fear.py`
	- Fear scenario with threat-vs-safe probabilistic path.
	- Appends row for `Fear`.

- `mdp_sadness.py`
	- Sadness scenario with model-change sensitivity in reward.
	- Appends row for `Sadness`.

- `mdp_shame.py`
	- Shame scenario with detour state and negative terminal branch.
	- Appends row for `Shame`.

### `Exp1_2/03_model_infer`

- `01_svm_infer.py`
	- Loads classifier training data + model appraisal vectors.
	- Trains linear SVM with sampled `C` values.
	- Writes model emotion probability tables:
		- `data/svm_free_0.0032_var.csv`
		- `data/svm_limit_0.014_var.csv`

- `pre/`
	- Auxiliary/intermediate folder from prior runs.

### `Exp1_2/04_statistical_analysis`

- `Exp1_free_analyse.py`
	- Free-rating condition analysis (Python reimplementation).
	- Mixed-effects model, CI bars, combined human-model plot, RMSE/scatter.
	- Saves plots such as `plots/exp1_free.png` and scatter output.

- `Exp2_limit_analyse.py`
	- Limited-rating condition analysis (Python reimplementation).
	- Same workflow for limit data and model outputs.
	- Saves plots such as `plots/exp2_limit.png` and `plots/exp2_scatter.png`.

- `Exp1_free_analyse.R`, `Exp2_limit_analyse.R`
	- Original R analysis scripts.

- `theme-publication.R`
	- Shared ggplot publication-style theme/fill helpers.

- `pre/`
	- Intermediate analysis artifacts.

### `Exp1_2/data`

- `human_free.csv`, `human_limit.csv`: human rating datasets.
- `classifier_train.csv`, `classifier_test.csv`: synthetic SVM train/test data.
- `model_result.csv`: appraisal vectors emitted by MDP simulations.
- `svm_free_0.0032_var.csv`, `svm_limit_0.014_var.csv`: model probability outputs.

### `Exp1_2/plots`

- Output figures from Python/R analysis scripts.

---

## `Exp3` (4 emotions)

### `Exp3/readme.md`

- Experiment-specific run notes and expected outputs.

### `Exp3/01_classifier`

- `01_analyze_human_data.py`
	- Reads `data/human_free_limit.csv`.
	- Computes precision stats for both free (`Emo.*`) and limit (`mc.*`) columns.

- `02_classifier.py`
	- Generates synthetic classifier train/test samples for:
		- `Anxiety`, `Despair`, `Irritation`, `Rage`
	- Writes:
		- `data/classifier_train.csv`
		- `data/classifier_test.csv`

- `03_determine_classifier_c.py`
	- Sweeps `C` and plots precision trend for Exp3 emotion space.

### `Exp3/02_mdp_model`

- `agent.py`
	- Shared RL appraisal agent (same core logic as Exp1_2).

- `01_get_model_data.py`
	- Initializes `data/model_result.csv`.
	- Runs all 4 emotion scenario scripts.

- `anxiety.py`, `despair.py`, `irritation.py`, `rage.py`
	- Emotion-specific transition/reward structures.
	- Each appends one appraisal row to `data/model_result.csv`.

### `Exp3/03_model_infer`

- `01_svm_infer.py`
	- Trains linear SVM over sampled `C` values and predicts probabilities on model appraisals.
	- Writes:
		- `data/svm_free_0.0013_var.csv`
		- `data/svm_limit_0.0034_var.csv`

### `Exp3/04_statistical_analysis`

- `Exp3_analyse.py`
	- Python analysis for both free and limit conditions in one script.
	- Mixed-effects models, CI bar plots, RMSE comparisons.
	- Saves:
		- `plots/exp3_free.png`
		- `plots/exp3_limit.png`

- `Exp3_analyse.R`
	- Original R version of Exp3 analyses.

### `Exp3/data`

- `human_free_limit.csv`: human data containing both free and limited formats.
- `classifier_train.csv`, `classifier_test.csv`: synthetic classifier data.
- `model_result.csv`: MDP appraisal vectors.
- `svm_free_0.0013_var.csv`, `svm_limit_0.0034_var.csv`: model predictions.

### `Exp3/plots`

- Output figures from Exp3 analyses.

---

## 4) Notes

- Both experiments include original R analysis scripts and Python equivalents.
- `__pycache__`, `.RData`, `.Rhistory`, and `pre/` folders/files are runtime or intermediate artifacts.
- For reproducibility, run each pipeline stage in order because later stages consume files produced earlier.

