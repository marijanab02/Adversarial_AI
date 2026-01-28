import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

dfs = []

# =========================================
# MODEL BEZ ADVERSARIAL TRAININGA
# =========================================
csv_files_no_adv = glob.glob("results/global_*.csv")

for f in csv_files_no_adv:
    df = pd.read_csv(f)
    filename = os.path.basename(f)

    if filename == "global_results.csv":
        epsilon = 0.0
    elif filename.startswith("global_results_fgsm_"):
        epsilon = float(
            filename.replace("global_results_fgsm_", "").replace(".csv", "")
        )
    else:
        continue

    df["epsilon"] = epsilon
    df["model"] = "Bez adversarial traininga"
    dfs.append(df)

# =========================================
# MODEL S ADVERSARIAL TRAININGOM
# =========================================
csv_files_adv = glob.glob("results_after_adversial_training/global_*.csv")

for f in csv_files_adv:
    df = pd.read_csv(f)
    filename = os.path.basename(f)

    if filename == "global_results_new.csv":
        epsilon = 0.0
    elif filename.startswith("global_results_fgsm_new_"):
        epsilon = float(
            filename.replace("global_results_fgsm_new_", "").replace(".csv", "")
        )
    else:
        continue

    df["epsilon"] = epsilon
    df["model"] = "S adversarial trainingom"
    dfs.append(df)

data = pd.concat(dfs).sort_values("epsilon")

print(data)


plt.figure()

for label, df_sub in data.groupby("model"):
    plt.plot(
        df_sub["epsilon"],
        df_sub["mAP50"],
        marker="o",
        label=label
    )

plt.xlabel("FGSM epsilon")
plt.ylabel("mAP50")
plt.title("mAP50 vs FGSM epsilon\nUsporedba modela")
plt.legend()
plt.grid(True)
plt.savefig("graphic_comparison_new/map_comparison.png", dpi=300)
plt.show()

# ==============================
# GRAF 2 – Precision vs epsilon
# ==============================

plt.figure()

for label, df_sub in data.groupby("model"):
    plt.plot(
        df_sub["epsilon"],
        df_sub["Precision"],
        marker="o",
        label=label
    )

plt.xlabel("FGSM epsilon")
plt.ylabel("Precision")
plt.title("Precision vs FGSM epsilon\nUsporedba modela")
plt.legend()
plt.grid(True)
plt.savefig("graphic_comparison_new/precision_comparison.png", dpi=300)
plt.show()



plt.figure()

for label, df_sub in data.groupby("model"):
    plt.plot(
        df_sub["epsilon"],
        df_sub["Recall"],
        marker="o",
        label=label
    )

plt.xlabel("FGSM epsilon")
plt.ylabel("Recall")
plt.title("Recall vs FGSM epsilon\nUsporedba modela")
plt.legend()
plt.grid(True)
plt.savefig("graphic_comparison_new/recall_comparison.png", dpi=300)
plt.show()
