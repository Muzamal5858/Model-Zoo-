"""
=============================================================
  MODEL ZOO: Bias vs Variance Explorer
  Topic: Bias-Variance Tradeoff in Machine Learning
  Author: ML Project
  Description:
    Trains multiple ML models on a noisy regression dataset,
    evaluates bias and variance empirically via bootstrap
    sampling, and plots learning curves + error decomposition.
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. DATASET GENERATION
# ─────────────────────────────────────────────
def generate_data(n=200, noise=0.5):
    """Generate noisy sine-wave regression data."""
    X = np.linspace(0, 10, n).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, noise, n)
    return X, y

X, y = generate_data(n=300, noise=0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ─────────────────────────────────────────────
# 2. MODEL ZOO DEFINITION
# ─────────────────────────────────────────────
models = {
    "Linear Regression\n(High Bias)": LinearRegression(),
    "Polynomial Deg-2\n(Low Bias)": Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("lin", LinearRegression())
    ]),
    "Polynomial Deg-10\n(High Variance)": Pipeline([
        ("poly", PolynomialFeatures(degree=10)),
        ("lin", LinearRegression())
    ]),
    "Ridge Regression\n(Balanced)": Pipeline([
        ("poly", PolynomialFeatures(degree=6)),
        ("ridge", Ridge(alpha=1.0))
    ]),
    "Decision Tree\n(High Variance)": DecisionTreeRegressor(max_depth=None),
    "Decision Tree\n(Pruned, Balanced)": DecisionTreeRegressor(max_depth=4),
    "KNN k=1\n(High Variance)": KNeighborsRegressor(n_neighbors=1),
    "KNN k=15\n(High Bias)": KNeighborsRegressor(n_neighbors=15),
}

# ─────────────────────────────────────────────
# 3. EMPIRICAL BIAS-VARIANCE DECOMPOSITION
#    Using bootstrap resampling
# ─────────────────────────────────────────────
def bias_variance_decomposition(model, X_train, y_train, X_test, y_test, n_bootstrap=50):
    """
    Empirically estimate bias^2, variance, and total MSE
    by training the model on multiple bootstrap samples.
    """
    predictions = []
    from sklearn.base import clone
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        Xb, yb = X_train[idx], y_train[idx]
        m = clone(model)
        m.fit(Xb, yb)
        predictions.append(m.predict(X_test))

    predictions = np.array(predictions)          # shape: (n_bootstrap, n_test)
    mean_pred   = predictions.mean(axis=0)       # average prediction per test point

    bias2    = np.mean((mean_pred - y_test) ** 2)
    variance = np.mean(predictions.var(axis=0))
    total_mse = bias2 + variance                 # noise excluded for clarity

    return round(bias2, 4), round(variance, 4), round(total_mse, 4)


print("\n" + "="*65)
print("  MODEL ZOO — Bias²  |  Variance  |  Total MSE (bootstrap, n=50)")
print("="*65)

bv_results = {}
for name, model in models.items():
    clean_name = name.replace("\n", " ")
    b2, var, mse = bias_variance_decomposition(model, X_train, y_train, X_test, y_test)
    bv_results[clean_name] = (b2, var, mse)
    print(f"  {clean_name:<38}  Bias²={b2:.4f}  Var={var:.4f}  MSE={mse:.4f}")

print("="*65)

# ─────────────────────────────────────────────
# 4. TRAIN / TEST MSE FOR EACH MODEL
# ─────────────────────────────────────────────
print("\n  TRAIN vs TEST MSE (single fit)")
print("-"*65)
fit_results = {}
for name, model in models.items():
    clean = name.replace("\n", " ")
    model.fit(X_train, y_train)
    tr_mse = mean_squared_error(y_train, model.predict(X_train))
    te_mse = mean_squared_error(y_test,  model.predict(X_test))
    fit_results[clean] = (round(tr_mse, 4), round(te_mse, 4))
    print(f"  {clean:<38}  Train={tr_mse:.4f}  Test={te_mse:.4f}")
print("-"*65)

# ─────────────────────────────────────────────
# 5. PLOTTING
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("#0d1117")
gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.35)

COLORS = {
    "train":    "#58a6ff",
    "test":     "#f78166",
    "bias":     "#79c0ff",
    "variance": "#ffa657",
    "mse":      "#56d364",
    "data":     "#8b949e",
    "fit":      "#ff7b72",
    "grid":     "#21262d",
    "text":     "#c9d1d9",
    "title":    "#e6edf3",
}

plt.rcParams.update({
    "text.color": COLORS["text"],
    "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "axes.facecolor": "#161b22",
    "axes.edgecolor": COLORS["grid"],
    "grid.color": COLORS["grid"],
    "axes.grid": True,
    "figure.facecolor": "#0d1117",
    "font.family": "monospace",
})

# ── PANEL 1: Raw data + model fits ─────────────────────────────
ax0 = fig.add_subplot(gs[0, :2])
ax0.scatter(X_train, y_train, color=COLORS["data"], s=10, alpha=0.5, label="Train data")
ax0.scatter(X_test,  y_test,  color=COLORS["fit"],  s=10, alpha=0.4, label="Test data")

X_plot = np.linspace(0, 10, 400).reshape(-1, 1)
highlight_models = {
    "Linear Regression (High Bias)": ("#79c0ff", "-"),
    "Polynomial Deg-10 (High Variance)": ("#ffa657", "--"),
    "Ridge Regression (Balanced)":       ("#56d364", "-"),
    "KNN k=1 (High Variance)":           ("#ff7b72", ":"),
}
for name, model in models.items():
    clean = name.replace("\n", " ")
    if clean in highlight_models:
        col, ls = highlight_models[clean]
        model.fit(X_train, y_train)
        ax0.plot(X_plot, model.predict(X_plot), color=col, lw=1.8,
                 linestyle=ls, label=clean.split("(")[0].strip(), alpha=0.9)

ax0.set_title("Model Fits on Noisy Data", color=COLORS["title"], fontsize=11, fontweight="bold")
ax0.legend(fontsize=7, loc="upper right", framealpha=0.2)
ax0.set_xlabel("X"); ax0.set_ylabel("y")

# ── PANEL 2: Bias² / Variance / MSE Bar Chart ──────────────────
ax1 = fig.add_subplot(gs[0, 2:])
short_labels = [n.split("(")[0].strip().replace(" ", "\n") for n in bv_results.keys()]
b2s   = [v[0] for v in bv_results.values()]
vars_ = [v[1] for v in bv_results.values()]
mses  = [v[2] for v in bv_results.values()]
x_idx = np.arange(len(short_labels))
w     = 0.25

ax1.bar(x_idx - w, b2s,   width=w, color=COLORS["bias"],     label="Bias²",    alpha=0.85)
ax1.bar(x_idx,     vars_, width=w, color=COLORS["variance"],  label="Variance", alpha=0.85)
ax1.bar(x_idx + w, mses,  width=w, color=COLORS["mse"],       label="MSE",      alpha=0.85)
ax1.set_xticks(x_idx)
ax1.set_xticklabels(short_labels, fontsize=6.5)
ax1.set_title("Bias² vs Variance (Bootstrap Decomposition)", color=COLORS["title"], fontsize=11, fontweight="bold")
ax1.legend(fontsize=8)
ax1.set_ylabel("Error")

# ── PANELS 3-10: Learning Curves per model ─────────────────────
axes_lc = [fig.add_subplot(gs[1 + i//4, i % 4]) for i in range(8)]

for idx, (name, model) in enumerate(models.items()):
    ax = axes_lc[idx]
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="neg_mean_squared_error"
    )
    tr_mean  = -train_scores.mean(axis=1)
    val_mean = -val_scores.mean(axis=1)
    tr_std   = train_scores.std(axis=1)
    val_std  = val_scores.std(axis=1)

    ax.plot(train_sizes, tr_mean,  color=COLORS["train"], lw=1.5, label="Train MSE")
    ax.plot(train_sizes, val_mean, color=COLORS["test"],  lw=1.5, label="Val MSE",  linestyle="--")
    ax.fill_between(train_sizes, tr_mean - tr_std,  tr_mean + tr_std,  alpha=0.1, color=COLORS["train"])
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color=COLORS["test"])

    title = name.replace("\n", " ")
    ax.set_title(title, fontsize=7.5, color=COLORS["title"], fontweight="bold")
    ax.set_xlabel("Training size", fontsize=7)
    ax.set_ylabel("MSE", fontsize=7)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.15)
    ax.set_ylim(bottom=0)

# ── PANEL 11: Train vs Test MSE summary ────────────────────────
ax_sum = fig.add_subplot(gs[3, :])
short2 = [n.split("(")[0].strip() for n in fit_results.keys()]
tr_vals = [v[0] for v in fit_results.values()]
te_vals = [v[1] for v in fit_results.values()]
x2      = np.arange(len(short2))

bars1 = ax_sum.bar(x2 - 0.2, tr_vals, 0.38, color=COLORS["train"], label="Train MSE", alpha=0.85)
bars2 = ax_sum.bar(x2 + 0.2, te_vals, 0.38, color=COLORS["test"],  label="Test MSE",  alpha=0.85)

for bar in bars1:
    ax_sum.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7, color=COLORS["train"])
for bar in bars2:
    ax_sum.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7, color=COLORS["test"])

ax_sum.set_xticks(x2)
ax_sum.set_xticklabels(short2, fontsize=8.5)
ax_sum.set_title("Train vs Test MSE — Full Model Comparison", color=COLORS["title"], fontsize=12, fontweight="bold")
ax_sum.legend(fontsize=9)
ax_sum.set_ylabel("Mean Squared Error")

fig.suptitle("MODEL ZOO: Bias–Variance Tradeoff Explorer",
             fontsize=16, color=COLORS["title"], fontweight="bold", y=0.995)

plt.savefig("/mnt/user-data/outputs/model_zoo_output.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("\n✅ Plot saved to: model_zoo_output.png")
print("\n📌 KEY TAKEAWAYS:")
print("  • Linear Regression / KNN k=15  → High Bias, Low Variance (Underfitting)")
print("  • Polynomial Deg-10 / KNN k=1   → Low Bias, High Variance (Overfitting)")
print("  • Ridge / Pruned Tree            → Balanced (Best generalization)")
print("  • Learning curves: train≈val gap reveals variance; high val error = bias")
