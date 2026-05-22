import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./temp/study_metrics.csv")

df["new_model_winrate"] = (
    df["arena_new_wins"] /
    (
        df["arena_new_wins"] +
        df["arena_prev_wins"] +
        df["arena_draws"]
    )
)

df["improvement_ratio"] = (
    (df["arena_new_wins"] - df["arena_prev_wins"]) /
    (
        df["arena_new_wins"] +
        df["arena_prev_wins"] + 1e-8
    )
)

df["sec_per_example"] = (
    df["duration_sec"] / df["num_examples"]
)

episodes_estimation = (
    df["num_examples"] / df["avg_episode_len"]
)

df["sec_per_episode"] = (
    df["duration_sec"] / episodes_estimation
)

df["cumulative_acceptance_rate"] = (
    df["accepted"].cumsum() /
    (np.arange(len(df)) + 1)
)

df["training_stability"] = (
    1.0 - abs(df["new_model_winrate"] - 0.5)
)

plt.figure(figsize=(10, 6))
plt.plot(df["iteration"], df["num_examples"], marker='o')
plt.xlabel("Iteration")
plt.ylabel("Training examples")
plt.title("Training examples over iterations")
plt.grid(True)
plt.savefig("./temp/examples_over_iterations.png")

plt.figure(figsize=(10, 6))
plt.plot(df["iteration"], df["avg_episode_len"], marker='o')
plt.xlabel("Iteration")
plt.ylabel("Average episode length")
plt.title("Average episode length evolution")
plt.grid(True)
plt.savefig("./temp/avg_episode_length.png")

plt.figure(figsize=(10, 6))
plt.plot(df["iteration"], df["arena_new_wins"], label="New model wins", marker='o')
plt.plot(df["iteration"], df["arena_prev_wins"], label="Previous model wins", marker='o')
plt.plot(df["iteration"], df["arena_draws"], label="Draws", marker='o')
plt.xlabel("Iteration")
plt.ylabel("Games")
plt.title("Arena comparison results")
plt.legend()
plt.grid(True)
plt.savefig("./temp/arena_results.png")

plt.figure(figsize=(10, 6))
plt.plot(df["iteration"], df["accepted"], marker='o')
plt.xlabel("Iteration")
plt.ylabel("Accepted")
plt.title("Model acceptance over iterations")
plt.grid(True)
plt.savefig("./temp/model_acceptance.png")

plt.figure(figsize=(10, 6))
plt.plot(df["iteration"], df["new_model_winrate"], marker='o')
plt.xlabel("Iteration")
plt.ylabel("Winrate")
plt.title("New model winrate evolution")
plt.grid(True)
plt.savefig("./temp/winrate_evolution.png")

plt.figure(figsize=(10, 6))
plt.plot(df["iteration"], df["improvement_ratio"], marker='o')
plt.xlabel("Iteration")
plt.ylabel("Improvement ratio")
plt.title("Relative improvement over previous model")
plt.grid(True)
plt.savefig("./temp/improvement_ratio.png")

plt.figure(figsize=(10, 6))
plt.plot(df["iteration"], df["sec_per_example"], marker='o')
plt.xlabel("Iteration")
plt.ylabel("Seconds per example")
plt.title("Training computational cost per example")
plt.grid(True)
plt.savefig("./temp/sec_per_example.png")

plt.figure(figsize=(10, 6))
plt.plot(df["iteration"], df["sec_per_episode"], marker='o')
plt.xlabel("Iteration")
plt.ylabel("Seconds per episode")
plt.title("Average episode computational cost")
plt.grid(True)
plt.savefig("./temp/sec_per_episode.png")

plt.figure(figsize=(10, 6))
plt.plot(df["iteration"], df["cumulative_acceptance_rate"], marker='o')
plt.xlabel("Iteration")
plt.ylabel("Acceptance rate")
plt.title("Cumulative model acceptance stability")
plt.grid(True)
plt.savefig("./temp/cumulative_acceptance_rate.png")

plt.figure(figsize=(10, 6))
plt.plot(df["iteration"], df["training_stability"], marker='o')
plt.xlabel("Iteration")
plt.ylabel("Stability score")
plt.title("Learning stability analysis")
plt.grid(True)
plt.savefig("./temp/training_stability.png")

correlation_columns = [
    "num_examples",
    "avg_episode_len",
    "arena_new_wins",
    "arena_prev_wins",
    "arena_draws",
    "accepted",
    "duration_sec",
    "new_model_winrate",
    "improvement_ratio",
    "sec_per_example",
    "sec_per_episode",
    "training_stability"
]

corr = df[correlation_columns].corr()

plt.figure(figsize=(12, 10))
plt.imshow(corr, interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Training metrics correlation matrix")
plt.tight_layout()
plt.savefig("./temp/correlation_matrix.png")

rolling = df["new_model_winrate"].rolling(window=2).mean()

plt.figure(figsize=(10, 6))
plt.plot(df["iteration"], rolling, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Rolling winrate")
plt.title("Rolling average winrate")
plt.grid(True)
plt.savefig("./temp/rolling_winrate.png")

plt.figure(figsize=(10, 6))
plt.scatter(df["duration_sec"], df["new_model_winrate"])
plt.xlabel("Training duration")
plt.ylabel("Winrate")
plt.title("Training duration vs winrate")
plt.grid(True)
plt.savefig("./temp/duration_vs_winrate.png")

plt.figure(figsize=(10, 6))
plt.scatter(df["num_examples"], df["new_model_winrate"])
plt.xlabel("Number of examples")
plt.ylabel("Winrate")
plt.title("Dataset size vs winrate")
plt.grid(True)
plt.savefig("./temp/examples_vs_winrate.png")

plt.figure(figsize=(10, 6))
plt.scatter(df["avg_episode_len"], df["new_model_winrate"])
plt.xlabel("Average episode length")
plt.ylabel("Winrate")
plt.title("Episode length vs winrate")
plt.grid(True)
plt.savefig("./temp/episode_length_vs_winrate.png")

plt.figure(figsize=(12, 10))
plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

plt.title("Training metrics correlation heatmap")
plt.tight_layout()
plt.savefig("./temp/correlation_heatmap_gradient.png")

boxplot_columns = [
    "num_examples",
    "avg_episode_len",
    "arena_new_wins",
    "arena_prev_wins",
    "arena_draws",
    "duration_sec",
    "new_model_winrate",
    "improvement_ratio",
    "sec_per_example",
    "sec_per_episode",
    "training_stability"
]

normalized_df = df[boxplot_columns].copy()

for col in boxplot_columns:
    min_val = normalized_df[col].min()
    max_val = normalized_df[col].max()

    if max_val != min_val:
        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    else:
        normalized_df[col] = 0

plt.figure(figsize=(14, 8))
plt.boxplot(
    [normalized_df[col].dropna() for col in boxplot_columns],
    labels=boxplot_columns,
    showmeans=True
)

plt.xticks(rotation=90)
plt.ylabel("Normalized value")
plt.title("Normalized distribution of training metrics")
plt.grid(True)
plt.tight_layout()
plt.savefig("./temp/metrics_boxplot.png")

print("Graphs saved in ./temp/")