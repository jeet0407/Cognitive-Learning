import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

FEATURE_COLUMNS = ['Suddenness', 'Goal_relevance', 'Conduciveness', 'Power', 'Urgency']
RANDOM_STATE = 7


def clip01(values):
    return np.clip(values, 0.0, 1.0)


def create_manual_gmm(profile, seed=RANDOM_STATE):
    means = np.asarray(profile['means'], dtype=float)
    scales = np.asarray(profile['scales'], dtype=float)
    weights = np.asarray(profile.get('weights'), dtype=float)

    gmm = GaussianMixture(
        n_components=len(weights),
        covariance_type='diag',
        random_state=seed,
    )
    gmm.weights_ = weights / weights.sum()
    gmm.means_ = means
    gmm.covariances_ = np.square(scales)
    gmm.precisions_ = 1.0 / gmm.covariances_
    gmm.precisions_cholesky_ = 1.0 / scales
    return gmm


def sample_emotion_profile(emotion, profile, n_samples, seed=RANDOM_STATE):
    gmm = create_manual_gmm(profile, seed=seed)
    samples, _ = gmm.sample(n_samples)
    samples = clip01(samples)
    frame = pd.DataFrame(samples, columns=FEATURE_COLUMNS)
    frame.insert(0, 'Emotion', emotion)
    return frame


def generate_gmm_dataset(emotion_profiles, samples_per_emotion, seed=RANDOM_STATE):
    frames = []
    for index, (emotion, profile) in enumerate(emotion_profiles.items()):
        frames.append(sample_emotion_profile(emotion, profile, samples_per_emotion, seed + index))
    data = pd.concat(frames, ignore_index=True)
    return data


def read_xy(data_file):
    data = pd.read_csv(data_file)
    data.columns = data.columns.str.strip()
    if 'Emotion' in data.columns:
        data['Emotion'] = data['Emotion'].astype(str).str.strip()
    X = clip01(data[FEATURE_COLUMNS].values.astype(float))
    y = data['Emotion'].values
    return X, y


def correct_class_probability_stats(probabilities, y_true, class_labels):
    class_to_index = {label: index for index, label in enumerate(class_labels)}
    correct = np.asarray(
        [probabilities[row_index][class_to_index[y_true[row_index]]] for row_index in range(len(y_true))],
        dtype=float,
    )
    return {
        'mean': float(correct.mean()),
        'var': float(correct.var()),
        'per_sample': correct,
    }


def alignment_score(mean_value, var_value, target_mean, target_var):
    mean_scale = max(target_mean, 0.05)
    var_scale = max(target_var, 0.05)
    return float(((mean_value - target_mean) / mean_scale) ** 2 + ((var_value - target_var) / var_scale) ** 2)


def evaluate_pipeline_config(X_train, y_train, X_test, y_test, c_value, gamma_value, kernel='rbf'):
    pipeline = Pipeline(
        [
            ('scaler', MinMaxScaler()),
            (
                'svc',
                SVC(
                    kernel=kernel,
                    C=float(c_value),
                    gamma=gamma_value,
                    probability=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)
    correct = (predicted == y_test).astype(float)
    stats = {
        'mean': float(correct.mean()),
        'var': float(correct.var()),
        'per_sample': correct,
    }
    return pipeline, stats


def run_rbf_grid_search(X_train, y_train, X_test, y_test, target_mean, target_var, c_grid, gamma_grid):
    pipeline = Pipeline(
        [
            ('scaler', MinMaxScaler()),
            ('svc', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)),
        ]
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline,
        param_grid={'svc__C': c_grid, 'svc__gamma': gamma_grid},
        scoring='accuracy',
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    rows = []
    for row_index in range(len(search.cv_results_['params'])):
        params = search.cv_results_['params'][row_index]
        c_value = float(params['svc__C'])
        gamma_value = params['svc__gamma']
        _, stats = evaluate_pipeline_config(X_train, y_train, X_test, y_test, c_value, gamma_value)
        rows.append(
            {
                'C': c_value,
                'gamma': gamma_value,
                'cv_accuracy': float(search.cv_results_['mean_test_score'][row_index]),
                'precision_mean': stats['mean'],
                'precision_var': stats['var'],
                'alignment_score': alignment_score(stats['mean'], stats['var'], target_mean, target_var),
            }
        )

    results = pd.DataFrame(rows)
    results['combined_score'] = results['alignment_score'] + 0.2 * (1.0 - results['cv_accuracy'])
    results = results.sort_values(['combined_score', 'alignment_score', 'cv_accuracy'], ascending=[True, True, False]).reset_index(drop=True)

    top_candidates = results.head(min(5, len(results))).copy()
    best = top_candidates.iloc[0].to_dict()

    tuning = {
        'best_C': float(best['C']),
        'best_gamma': best['gamma'],
        'best_cv_accuracy': float(best['cv_accuracy']),
        'precision_mean': float(best['precision_mean']),
        'precision_var': float(best['precision_var']),
        'c_mean': float(top_candidates['C'].mean()),
        'c_var': float(top_candidates['C'].var(ddof=0)),
        'gamma_mode': str(top_candidates['gamma'].mode().iloc[0]),
        'top_candidates': top_candidates.to_dict(orient='records'),
    }
    return tuning, results


def save_tuning(path, free_tuning, limit_tuning):
    payload = {'free': free_tuning, 'limit': limit_tuning}
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def load_tuning(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def sampled_c_values(mean_value, variance_value, sample_size, seed_offset=0):
    rng = np.random.default_rng(RANDOM_STATE + seed_offset)
    std_value = float(np.sqrt(max(variance_value, 1e-8)))
    values = rng.normal(float(mean_value), std_value, int(sample_size))
    return np.clip(values, 1e-5, None)


def write_probability_table(output_file, story_labels, target_names, sample_values, predictor_factory):
    rows = []
    for c_value in sample_values:
        predictor = predictor_factory(float(c_value))
        for story_label, probabilities in zip(story_labels, predictor):
            for emotion, prob in zip(target_names, probabilities):
                rows.append([float(c_value), story_label, emotion, float(prob)])

    pd.DataFrame(rows, columns=['C', 'Story', 'Emotion', 'Val']).to_csv(output_file, index=False)


def aggregated_model_distribution(model_file):
    data = pd.read_csv(model_file)
    return data.groupby(['Story', 'Emotion'], as_index=False)['Val'].mean()


def normalized_human_distribution(human_long):
    grouped = human_long.groupby(['Story', 'Emotion'], as_index=False)['Val'].mean()
    grouped['Val'] = grouped.groupby('Story')['Val'].transform(lambda values: values / values.sum())
    return grouped


def compare_distributions(human_distribution, model_distribution):
    merged = human_distribution.merge(model_distribution, on=['Story', 'Emotion'], suffixes=('_human', '_model'))
    rmse = float(np.sqrt(np.mean((merged['Val_human'] - merged['Val_model']) ** 2)))
    r2 = float(r2_score(merged['Val_human'], merged['Val_model']))
    return rmse, r2, merged


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def stratified_subsample(X, y, max_per_class, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    keep_indices = []
    labels = np.asarray(y)
    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        sample_size = min(len(label_indices), int(max_per_class))
        keep_indices.extend(rng.choice(label_indices, size=sample_size, replace=False).tolist())
    keep_indices = np.asarray(sorted(keep_indices))
    return X[keep_indices], labels[keep_indices]
