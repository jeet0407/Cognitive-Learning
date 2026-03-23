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
# Load Human Data
# ---------------------------
data = pd.read_csv(os.path.join(exp_root, "data", "human_limit.csv"), sep=";")

# Pivot (wide → long)
data = data.melt(
    id_vars=["ID", "Story"],
    value_vars=[col for col in data.columns if col.startswith("Em.")],
    var_name="Emotion",
    value_name="Val"
)

data["Emotion"] = data["Emotion"].str.replace("Em.", "")

# ---------------------------
# Mixed Effects Model
# ---------------------------
model = smf.mixedlm("Val ~ Story * Emotion", data, groups=data["ID"])
result = model.fit()
print(result.summary())

# ---------------------------
# Human Aggregation + CI
# ---------------------------
grouped = data.groupby(["Story", "Emotion"])

data_h = grouped["Val"].agg(['mean', 'std']).reset_index()
data_h.rename(columns={"mean": "Val", "std": "sd"}, inplace=True)

n = data["ID"].nunique()

data_h["SE"] = data_h["sd"] / np.sqrt(n)
data_h["CI_lower"] = np.maximum(data_h["Val"] - 1.96 * data_h["SE"], 0)
data_h["CI_upper"] = np.minimum(data_h["Val"] + 1.96 * data_h["SE"], 1)
data_h["Source"] = "Human"

# ---------------------------
# Model Data
# ---------------------------
data_m = pd.read_csv(os.path.join(exp_root, "data", "svm_limit_0.014_var.csv"))

data_m = data_m.groupby(["Story", "Emotion"])["Val"].mean().reset_index()
data_m["CI_lower"] = data_m["Val"]
data_m["CI_upper"] = data_m["Val"]
data_m["Source"] = "Model"

# ---------------------------
# Combine Data
# ---------------------------
data_b = pd.concat([data_h, data_m])

# Order categories
order = ["Happiness","Joy","Pride","Boredom","Fear","Sadness","Shame"]

data_b["Story"] = pd.Categorical(data_b["Story"], categories=order, ordered=True)
data_b["Emotion"] = pd.Categorical(data_b["Emotion"], categories=order, ordered=True)

# ---------------------------
# Remove zero values (like df_nonzero in R)
# ---------------------------
df_nonzero = data_b[data_b["Val"] != 0]

# ---------------------------
# Plot (Bar + Error bars)
# ---------------------------
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

# Add error bars only for human + non-zero
for ax, story in zip(g.axes.flatten(), order):
    subset = df_nonzero[(df_nonzero["Story"] == story) & (df_nonzero["Source"] == "Human")]

    if len(subset) > 0:
        ax.errorbar(
            x=np.arange(len(subset)),
            y=subset["Val"],
            yerr=[subset["Val"] - subset["CI_lower"], subset["CI_upper"] - subset["Val"]],
            fmt='none',
            capsize=3
        )

plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(exp_root, "plots", "exp2_limit.png"), dpi=300)

# ---------------------------
# Long Form Comparison
# ---------------------------
data_bl = data.groupby(["Story", "Emotion"])["Val"].mean().reset_index()
data_bl["Val"] = data_bl.groupby("Story")["Val"].transform(lambda x: x / x.sum())

data_bl = data_bl.merge(data_m, on=["Story", "Emotion"], suffixes=("_x", "_y"))
data_bl["d"] = (data_bl["Val_x"] - data_bl["Val_y"])**2

# RMSE
rmse = np.sqrt(data_bl["d"].mean())
print("RMSE:", rmse)

# ---------------------------
# Scatter Plot
# ---------------------------
sns.regplot(data=data_bl, x="Val_y", y="Val_x")
plt.xlim(0, 0.5)
plt.ylim(0, 0.5)
plt.savefig(os.path.join(exp_root, "plots", "exp2_scatter.png"))

# ---------------------------
# Linear Regression Summary
# ---------------------------
lm_model = smf.ols("Val_x ~ Val_y", data=data_bl).fit()
print(lm_model.summary())