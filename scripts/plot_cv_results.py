import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# 学習時の結果をpickleで保存している場合
with open("results_cv.pkl", "rb") as f:
    results = pickle.load(f)

# Accuracyの棒グラフ
plt.figure(figsize=(7, 4))
sns.barplot(x=list(range(1, len(results['cv_scores']) + 1)), y=results['cv_scores'])
plt.title("Accuracy for each fold")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# 各foldごとの混同行列ヒートマップ
for i, cm in enumerate(results['conf_matrices']):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
    plt.title(f"Confusion Matrix: Fold {i + 1}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# f1-score, precision, recallの推移
f1s, precs, recs = [], [], []
for r in results['cv_reports']:
    f1s.append((r['0']['f1-score'] + r['1']['f1-score'] + r['2']['f1-score']) / 3)
    precs.append((r['0']['precision'] + r['1']['precision'] + r['2']['precision']) / 3)
    recs.append((r['0']['recall'] + r['1']['recall'] + r['2']['recall']) / 3)

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(f1s) + 1), f1s, label="F1-score", marker='o')
plt.plot(range(1, len(precs) + 1), precs, label="Precision", marker='s')
plt.plot(range(1, len(recs) + 1), recs, label="Recall", marker='^')
plt.title("Macro average per fold")
plt.xlabel("Fold")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
plt.show()