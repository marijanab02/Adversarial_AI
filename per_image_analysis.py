import pandas as pd
import matplotlib.pyplot as plt
import glob

files = {

    0.0:   "results_after_adversial_training/test_results_per_image_new.csv",
    0.005: "results_after_adversial_training/test_results_per_image_fgsm_new_0.005.csv",
    0.01:   "results_after_adversial_training/test_results_per_image_fgsm_new_0.01.csv",
    0.02:  "results_after_adversial_training/test_results_per_image_fgsm_new_0.02.csv",
    0.03:  "results_after_adversial_training/test_results_per_image_fgsm_new_0.03.csv",
    0.05:  "results_after_adversial_training/test_results_per_image_fgsm_new_0.05.csv",
    0.1:   "results_after_adversial_training/test_results_per_image_fgsm_new_0.1.csv",
}


epsilons = []
false_negatives = []
false_positives = []
true_positives = []

for eps, path in files.items():
    df = pd.read_csv(path)
    epsilons.append(eps)
    false_negatives.append(df["false_negatives"].sum())
    true_positives.append(df["true_positives"].sum())
    false_positives.append(df["false_positives"].sum())

# ==============================
# GRAF – False Negatives
# ==============================

plt.figure()
plt.plot(epsilons, false_negatives, marker="o")
plt.xlabel("FGSM epsilon")
plt.ylabel("Total False Negatives")
plt.title("False Negatives vs FGSM epsilon")
plt.grid(True)
plt.savefig("graphic_comparison_new/false_negatives_vs_epsilon.png", dpi=300)
plt.show()

# ==============================
# GRAF – False Positives
# ==============================

plt.figure()
plt.plot(epsilons, false_positives, marker="o")
plt.xlabel("FGSM epsilon")
plt.ylabel("Total False Positives")
plt.title("False Positives vs FGSM epsilon")
plt.grid(True)
plt.savefig("graphic_comparison_new/false_positives_vs_epsilon.png", dpi=300)
plt.show()

plt.figure()
plt.plot(epsilons, false_negatives, marker="o")
plt.xlabel("FGSM epsilon")
plt.ylabel("Total True Positives")
plt.title("True Positives vs FGSM epsilon")
plt.grid(True)

plt.savefig("graphic_comparison_new/true_positives_vs_epsilon.png", dpi=300)
plt.show()
