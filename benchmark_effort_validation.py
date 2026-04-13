import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev


EXP1_PIPELINE = [
    "01_classifier/01_analyze_human_data.py",
    "01_classifier/02_classifier.py",
    "01_classifier/03_determine_classifier_c.py",
    "02_mdp_model/01_get_model_data.py",
    "03_model_infer/01_svm_infer.py",
    "04_statistical_analysis/Exp1_free_analyse.py",
    "04_statistical_analysis/Exp2_limit_analyse.py",
]

EXP3_PIPELINE = [
    "01_classifier/01_analyze_human_data.py",
    "01_classifier/02_classifier.py",
    "01_classifier/03_determine_classifier_c.py",
    "02_mdp_model/01_get_model_data.py",
    "03_model_infer/01_svm_infer.py",
    "04_statistical_analysis/Exp3_analyse.py",
]


def run_script(py_exe, cwd, script, env, log_dir):
    script_safe = script.replace("/", "_").replace(".", "_")
    stdout_path = log_dir / f"{script_safe}.stdout.log"
    stderr_path = log_dir / f"{script_safe}.stderr.log"
    t0 = time.perf_counter()
    proc = subprocess.run(
        [py_exe, script],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    dt = time.perf_counter() - t0
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed: {cwd.name}/{script} exit={proc.returncode}\n"
            f"stderr:\n{proc.stderr[:2000]}"
        )
    return dt, proc.stdout


def parse_exp1_metrics(free_stdout, limit_stdout):
    rmse_re = re.compile(r"RMSE:\s*([0-9eE\.\-]+)")
    r2_re = re.compile(r"R-squared:\s*([0-9eE\.\-]+)")
    free_rmse = float(rmse_re.search(free_stdout).group(1))
    limit_rmse = float(rmse_re.search(limit_stdout).group(1))
    r2_match = r2_re.search(limit_stdout)
    limit_r2 = float(r2_match.group(1)) if r2_match else None
    return free_rmse, limit_rmse, limit_r2


def parse_exp3_metrics(stdout):
    free_re = re.compile(r"FREE RMSE:\s*([0-9eE\.\-]+)")
    limit_re = re.compile(r"LIMIT RMSE:\s*([0-9eE\.\-]+)")
    free_rmse = float(free_re.search(stdout).group(1))
    limit_rmse = float(limit_re.search(stdout).group(1))
    return free_rmse, limit_rmse


def run_experiment(py_exe, exp_dir, pipeline, env, seed, variant, root_log_dir):
    run_log_dir = root_log_dir / f"{variant}_seed_{seed}" / exp_dir.name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    total = 0.0
    model_gen = None
    outputs = {}
    for script in pipeline:
        dt, out = run_script(py_exe, exp_dir, script, env, run_log_dir)
        total += dt
        outputs[script] = out
        if script == "02_mdp_model/01_get_model_data.py":
            model_gen = dt
    return total, model_gen, outputs


def safe_mean_std(vals):
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), stdev(vals)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(per_seed_rows):
    metrics = [
        "exp1_free_rmse",
        "exp1_limit_rmse",
        "exp1_limit_r2",
        "exp3_free_rmse",
        "exp3_limit_rmse",
    ]
    variants = sorted(set(r["variant"] for r in per_seed_rows))
    summary_rows = []
    by_variant_metric = {}
    for variant in variants:
        by_variant_metric[variant] = {}
        rows_v = [r for r in per_seed_rows if r["variant"] == variant]
        for m in metrics:
            vals = [r[m] for r in rows_v if r[m] is not None]
            mu, sd = safe_mean_std(vals)
            by_variant_metric[variant][m] = (mu, sd)
            summary_rows.append(
                {
                    "variant": variant,
                    "metric": m,
                    "mean": mu,
                    "std": sd,
                    "n": len(vals),
                }
            )

    if "baseline" in by_variant_metric and "effort" in by_variant_metric:
        for m in metrics:
            mu_b = by_variant_metric["baseline"][m][0]
            mu_e = by_variant_metric["effort"][m][0]
            if mu_b is not None and mu_e is not None:
                summary_rows.append(
                    {
                        "variant": "delta_effort_minus_baseline",
                        "metric": m,
                        "mean": mu_e - mu_b,
                        "std": None,
                        "n": None,
                    }
                )

    runtime_metrics = [
        "exp1_model_gen_s",
        "exp1_full_s",
        "exp3_model_gen_s",
        "exp3_full_s",
    ]
    runtime_rows = []
    by_variant_runtime = {}
    for variant in variants:
        by_variant_runtime[variant] = {}
        rows_v = [r for r in per_seed_rows if r["variant"] == variant]
        for m in runtime_metrics:
            vals = [r[m] for r in rows_v if r[m] is not None]
            mu, sd = safe_mean_std(vals)
            by_variant_runtime[variant][m] = (mu, sd)
            runtime_rows.append(
                {
                    "variant": variant,
                    "runtime_metric": m,
                    "mean_s": mu,
                    "std_s": sd,
                    "n": len(vals),
                }
            )
    if "baseline" in by_variant_runtime and "effort" in by_variant_runtime:
        for m in runtime_metrics:
            mu_b = by_variant_runtime["baseline"][m][0]
            mu_e = by_variant_runtime["effort"][m][0]
            if mu_b is not None and mu_e is not None:
                runtime_rows.append(
                    {
                        "variant": "delta_effort_minus_baseline",
                        "runtime_metric": m,
                        "mean_s": mu_e - mu_b,
                        "std_s": None,
                        "n": None,
                    }
                )

    return summary_rows, runtime_rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seeds",
        default="7,13,23,42,99,123,202,404,777,1024",
        help="Comma-separated integer seeds",
    )
    p.add_argument(
        "--output-dir",
        default="run_logs_effort_validation",
    )
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    root = Path(__file__).resolve().parent
    out_dir = root / args.output_dir
    raw_log_dir = out_dir / "raw_logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    py_exe = sys.executable
    per_seed_rows = []
    for seed in seeds:
        for variant in ("baseline", "effort"):
            env = os.environ.copy()
            env["RUN_SEED"] = str(seed)
            env["APPRAISAL_VARIANT"] = variant

            exp1_dir = root / "Exp1_2"
            exp3_dir = root / "Exp3"

            exp1_total, exp1_model_gen, exp1_out = run_experiment(
                py_exe, exp1_dir, EXP1_PIPELINE, env, seed, variant, raw_log_dir
            )
            exp3_total, exp3_model_gen, exp3_out = run_experiment(
                py_exe, exp3_dir, EXP3_PIPELINE, env, seed, variant, raw_log_dir
            )

            exp1_free_rmse, exp1_limit_rmse, exp1_limit_r2 = parse_exp1_metrics(
                exp1_out["04_statistical_analysis/Exp1_free_analyse.py"],
                exp1_out["04_statistical_analysis/Exp2_limit_analyse.py"],
            )
            exp3_free_rmse, exp3_limit_rmse = parse_exp3_metrics(
                exp3_out["04_statistical_analysis/Exp3_analyse.py"]
            )

            per_seed_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "exp1_free_rmse": exp1_free_rmse,
                    "exp1_limit_rmse": exp1_limit_rmse,
                    "exp1_limit_r2": exp1_limit_r2,
                    "exp3_free_rmse": exp3_free_rmse,
                    "exp3_limit_rmse": exp3_limit_rmse,
                    "exp3_limit_r2": None,
                    "exp1_model_gen_s": exp1_model_gen,
                    "exp1_full_s": exp1_total,
                    "exp3_model_gen_s": exp3_model_gen,
                    "exp3_full_s": exp3_total,
                }
            )
            print(f"Completed seed={seed}, variant={variant}")

    per_seed_file = out_dir / "per_seed_results.csv"
    write_csv(
        per_seed_file,
        per_seed_rows,
        fieldnames=[
            "seed",
            "variant",
            "exp1_free_rmse",
            "exp1_limit_rmse",
            "exp1_limit_r2",
            "exp3_free_rmse",
            "exp3_limit_rmse",
            "exp3_limit_r2",
            "exp1_model_gen_s",
            "exp1_full_s",
            "exp3_model_gen_s",
            "exp3_full_s",
        ],
    )

    summary_rows, runtime_rows = aggregate(per_seed_rows)
    write_csv(
        out_dir / "summary_mean_std.csv",
        summary_rows,
        fieldnames=["variant", "metric", "mean", "std", "n"],
    )
    write_csv(
        out_dir / "runtime_summary.csv",
        runtime_rows,
        fieldnames=["variant", "runtime_metric", "mean_s", "std_s", "n"],
    )

    print(f"Wrote: {per_seed_file}")
    print(f"Wrote: {out_dir / 'summary_mean_std.csv'}")
    print(f"Wrote: {out_dir / 'runtime_summary.csv'}")


if __name__ == "__main__":
    main()
