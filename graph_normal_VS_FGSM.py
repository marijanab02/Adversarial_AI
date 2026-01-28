import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

normal_global_csv = "global_results.csv"
fgsm_global_csv = "global_results_fgsm.csv"

normal_per_image_csv = "test_results_per_image.csv"
fgsm_per_image_csv = "test_results_per_image_fgsm.csv"

df_global_normal = pd.read_csv(normal_global_csv)
df_global_fgsm = pd.read_csv(fgsm_global_csv)

metrics = ["mAP50", "mAP50-95", "Precision", "Recall"]

plt.figure(figsize=(10,6))
for metric in metrics:
    plt.bar(metric + "_normal", df_global_normal[metric][0], color='skyblue', label='Normal' if metric=="mAP50" else "")
    plt.bar(metric + "_fgsm", df_global_fgsm[metric][0], color='salmon', label='FGSM' if metric=="mAP50" else "")

plt.ylabel("Vrijednost")
plt.title("Globalni rezultati: Normal vs FGSM")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("global_comparison.png")
plt.show()

df_normal = pd.read_csv(normal_per_image_csv)
df_fgsm = pd.read_csv(fgsm_per_image_csv)

# Spoji po imenu slike
df_compare = df_normal[["image", "ground_truth_humans", "predicted_humans"]].merge(
    df_fgsm[["image", "predicted_humans"]], on="image", suffixes=("_normal", "_fgsm")
)

plt.figure(figsize=(12,6))
sns.scatterplot(x="ground_truth_humans", y="predicted_humans_normal", data=df_compare, label="Normal", color='blue', s=60)
sns.scatterplot(x="ground_truth_humans", y="predicted_humans_fgsm", data=df_compare, label="FGSM", color='red', s=60)
plt.plot([0, df_compare["ground_truth_humans"].max()], [0, df_compare["ground_truth_humans"].max()], "k--", label="Ideal")
plt.xlabel("Broj ljudi (ground truth)")
plt.ylabel("Predviđeni broj ljudi")
plt.title("Per-image predikcije: Normal vs FGSM")
plt.legend()
plt.tight_layout()
plt.savefig("per_image_comparison.png")
plt.show()
