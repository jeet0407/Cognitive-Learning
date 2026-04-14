import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from common.five_dim_pipeline import (
    aggregated_model_distribution,
    compare_distributions,
    normalized_human_distribution,
)


def load_exp1_human_free(exp_root):
    data = pd.read_csv(exp_root / 'data' / 'human_free.csv', sep=';')
    data = data.melt(
        id_vars=['ID', 'Story'],
        value_vars=[col for col in data.columns if col.startswith('Emo.')],
        var_name='Emotion',
        value_name='Val',
    )
    data['Emotion'] = data['Emotion'].str.replace('Emo.', '', regex=False)
    return data


def load_exp1_human_limit(exp_root):
    data = pd.read_csv(exp_root / 'data' / 'human_limit.csv', sep=';')
    data = data.melt(
        id_vars=['ID', 'Story'],
        value_vars=[col for col in data.columns if col.startswith('Em.')],
        var_name='Emotion',
        value_name='Val',
    )
    data['Emotion'] = data['Emotion'].str.replace('Em.', '', regex=False)
    return data


def load_exp3_human_free(exp_root):
    data = pd.read_csv(exp_root / 'data' / 'human_free_limit.csv', sep=';')
    data = data.melt(
        id_vars=['ID', 'Story'],
        value_vars=[col for col in data.columns if col.startswith('Emo.')],
        var_name='Emotion',
        value_name='Val',
    )
    data['Emotion'] = data['Emotion'].str.replace('Emo.', '', regex=False)
    return data


def load_exp3_human_limit(exp_root):
    data = pd.read_csv(exp_root / 'data' / 'human_free_limit.csv', sep=';')
    data = data.melt(
        id_vars=['ID', 'Story'],
        value_vars=[col for col in data.columns if col.startswith('mc.')],
        var_name='Emotion',
        value_name='Val',
    )
    data['Emotion'] = data['Emotion'].str.replace('mc.', '', regex=False)
    return data


def evaluate_condition(exp_root, human_loader, baseline_file, optimized_file, label):
    human_distribution = normalized_human_distribution(human_loader(exp_root))
    baseline_distribution = aggregated_model_distribution(exp_root / 'data' / baseline_file)
    optimized_distribution = aggregated_model_distribution(exp_root / 'data' / optimized_file)

    baseline_rmse, baseline_r2, _ = compare_distributions(human_distribution, baseline_distribution)
    optimized_rmse, optimized_r2, _ = compare_distributions(human_distribution, optimized_distribution)

    return {
        'condition': label,
        'baseline_rmse': baseline_rmse,
        'baseline_r2': baseline_r2,
        'optimized_rmse': optimized_rmse,
        'optimized_r2': optimized_r2,
    }


def main():
    exp1_root = ROOT_DIR / 'Exp1_2'
    exp3_root = ROOT_DIR / 'Exp3'

    rows = [
        evaluate_condition(exp1_root, load_exp1_human_free, 'svm_free_linear_baseline.csv', 'svm_free_rbf_opt.csv', 'Exp1 free'),
        evaluate_condition(exp1_root, load_exp1_human_limit, 'svm_limit_linear_baseline.csv', 'svm_limit_rbf_opt.csv', 'Exp2 limit'),
        evaluate_condition(exp3_root, load_exp3_human_free, 'svm_free_linear_baseline.csv', 'svm_free_rbf_opt.csv', 'Exp3 free'),
        evaluate_condition(exp3_root, load_exp3_human_limit, 'svm_limit_linear_baseline.csv', 'svm_limit_rbf_opt.csv', 'Exp3 limit'),
    ]

    metrics = pd.DataFrame(rows)
    print(metrics.round(6).to_string(index=False))


if __name__ == '__main__':
    main()
