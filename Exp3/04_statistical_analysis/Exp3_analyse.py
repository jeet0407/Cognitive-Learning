import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# ---------------------------
# Paths
# ---------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
exp_root = os.path.abspath(os.path.join(script_dir, ".."))

# ---------------------------
# Helper function
# ---------------------------
def compute_ci(df):
    grouped = df.groupby(["Story", "Emotion"])
    out = grouped["Val"].agg(['mean', 'std']).reset_index()
    out.rename(columns={"mean": "Val", "std": "sd"}, inplace=True)

    n = df["ID"].nunique()
    out["SE"] = out["sd"] / np.sqrt(n)
    out["CI_lower"] = np.maximum(out["Val"] - 1.96 * out["SE"], 0)
    out["CI_upper"] = np.minimum(out["Val"] + 1.96 * out["SE"], 1)

    return out[["Story", "Emotion", "Val", "CI_lower", "CI_upper"]]

# ===========================
# 🔵 FREE RATING
# ===========================
data = pd.read_csv(os.path.join(exp_root, "data", "human_free_limit.csv"), sep=";")

data = data.melt(
    id_vars=["ID", "Story"],
    value_vars=[c for c in data.columns if c.startswith("Emo.")],
    var_name="Emotion",
    value_name="Val"
)

data["Emotion"] = data["Emotion"].str.replace("Emo.", "")

# Mixed model
model = smf.mixedlm("Val ~ Story * Emotion", data, groups=data["ID"])
print(model.fit().summary())

# Human CI
data_h = compute_ci(data)
data_h["Source"] = "Human"

# Model data
data_m = pd.read_csv(os.path.join(exp_root, "data", "svm_free_0.0013_var.csv"))
data_m = data_m.groupby(["Story", "Emotion"])["Val"].mean().reset_index()
data_m["CI_lower"] = data_m["Val"]
data_m["CI_upper"] = data_m["Val"]
data_m["Source"] = "Model"

# Combine
data_b = pd.concat([data_h, data_m])

# Plot
g = sns.catplot(
    data=data_b,
    x="Emotion",
    y="Val",
    hue="Source",
    col="Story",
    kind="bar",
    col_wrap=1,
    height=3,
    aspect=2
)

for ax, story in zip(g.axes.flatten(), data_b["Story"].unique()):
    subset = data_b[(data_b["Story"] == story) & (data_b["Source"] == "Human")]
    ax.errorbar(
        x=np.arange(len(subset)),
        y=subset["Val"],
        yerr=[subset["Val"] - subset["CI_lower"], subset["CI_upper"] - subset["Val"]],
        fmt='none',
        capsize=3
    )

plt.ylim(0, 0.6)
plt.tight_layout()
plt.savefig(os.path.join(exp_root, "plots", "exp3_free.png"), dpi=300)

# RMSE
data_bl = data.groupby(["Story", "Emotion"])["Val"].mean().reset_index()
data_bl["Val"] = data_bl.groupby("Story")["Val"].transform(lambda x: x / x.sum())

data_bl = data_bl.merge(data_m, on=["Story", "Emotion"], suffixes=("_x", "_y"))
data_bl["d"] = (data_bl["Val_x"] - data_bl["Val_y"])**2

print("FREE RMSE:", np.sqrt(data_bl["d"].mean()))

# ===========================
# 🔴 LIMIT RATING
# ===========================
data2 = pd.read_csv(os.path.join(exp_root, "data", "human_free_limit.csv"), sep=";")

data2 = data2.melt(
    id_vars=["ID", "Story"],
    value_vars=[c for c in data2.columns if c.startswith("mc")],
    var_name="Emotion",
    value_name="Val"
)

data2["Emotion"] = data2["Emotion"].str.replace("mc.", "")

# Mixed model
model2 = smf.mixedlm("Val ~ Story * Emotion", data2, groups=data2["ID"])
print(model2.fit().summary())

# Human CI
data_h2 = compute_ci(data2)
data_h2["Source"] = "Human"

# Model data
data_m2 = pd.read_csv(os.path.join(exp_root, "data", "svm_limit_0.0034_var.csv"))
data_m2 = data_m2.groupby(["Story", "Emotion"])["Val"].mean().reset_index()
data_m2["CI_lower"] = data_m2["Val"]
data_m2["CI_upper"] = data_m2["Val"]
data_m2["Source"] = "Model"

# Combine
data_b2 = pd.concat([data_h2, data_m2])

# Plot
g2 = sns.catplot(
    data=data_b2,
    x="Emotion",
    y="Val",
    hue="Source",
    col="Story",
    kind="bar",
    col_wrap=1,
    height=3,
    aspect=2
)

for ax, story in zip(g2.axes.flatten(), data_b2["Story"].unique()):
    subset = data_b2[(data_b2["Story"] == story) & (data_b2["Source"] == "Human")]
    ax.errorbar(
        x=np.arange(len(subset)),
        y=subset["Val"],
        yerr=[subset["Val"] - subset["CI_lower"], subset["CI_upper"] - subset["Val"]],
        fmt='none',
        capsize=3
    )

plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(exp_root, "plots", "exp3_limit.png"), dpi=300)

# RMSE
data_bl2 = data2.groupby(["Story", "Emotion"])["Val"].mean().reset_index()
data_bl2["Val"] = data_bl2.groupby("Story")["Val"].transform(lambda x: x / x.sum())

data_bl2 = data_bl2.merge(data_m2, on=["Story", "Emotion"], suffixes=("_x", "_y"))
data_bl2["d"] = (data_bl2["Val_x"] - data_bl2["Val_y"])**2

print("LIMIT RMSE:", np.sqrt(data_bl2["d"].mean()))