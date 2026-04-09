import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# FILE PATHS (epsilon -> CSV)
# ==============================
files = {
    0.0:   "results_after_adversial_training/test_results_per_image_new.csv",
    0.005: "results_after_adversial_training/test_results_per_image_fgsm_new_0.005.csv",
    0.01:  "results_after_adversial_training/test_results_per_image_fgsm_new_0.01.csv",
    0.02:  "results_after_adversial_training/test_results_per_image_fgsm_new_0.02.csv",
    0.03:  "results_after_adversial_training/test_results_per_image_fgsm_new_0.03.csv",
    0.05:  "results_after_adversial_training/test_results_per_image_fgsm_new_0.05.csv",
    0.1:   "results_after_adversial_training/test_results_per_image_fgsm_new_0.1.csv",
}

# ==============================
# DATA COLLECTION
# ==============================
epsilons = []
false_negatives = []
false_positives = []
true_positives = []

for eps, path in files.items():
    df = pd.read_csv(path)

    epsilons.append(eps)
    false_negatives.append(df["false_negatives"].sum())
    false_positives.append(df["false_positives"].sum())
    true_positives.append(df["true_positives"].sum())

# ==============================
# SORT BY EPSILON (IMPORTANT)
# ==============================
epsilons, false_negatives, false_positives, true_positives = zip(
    *sorted(zip(epsilons, false_negatives, false_positives, true_positives))
)

# ==============================
# PLOT – ALL METRICS IN ONE GRAPH
# ==============================
plt.figure(figsize=(8, 5))

plt.plot(epsilons, false_negatives, marker="o", linewidth=2, label="False Negatives")
plt.plot(epsilons, false_positives, marker="o", linewidth=2, label="False Positives")
plt.plot(epsilons, true_positives, marker="o", linewidth=2, label="True Positives")

plt.xlabel("FGSM epsilon")
plt.ylabel("Total detections")
plt.title("Detection Metrics vs FGSM epsilon")

plt.legend()
plt.grid(True)

# ==============================
# SAVE + SHOW
# ==============================
plt.savefig("graphic_comparison_new/all_metrics_vs_epsilon.png", dpi=300, bbox_inches="tight")
plt.show()