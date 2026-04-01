import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd

def evaluate_model(model, X_test, y_test, feature_names):

    y_probs = model.predict_proba(X_test)[:, 1]

    auc_score = roc_auc_score(y_test, y_probs)
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)

    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importances, palette='viridis')
    plt.title('Top 10 features')
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', label=f'AP = {average_precision_score(y_test, y_probs):.2f}')
    plt.xlabel('Recall (How many frauds)')
    plt.ylabel('Precision (Client wrongly blocked)')
    plt.title('Precision-Recall curve')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Linia losowa
    plt.xlabel('False Positive Rate (Unfairly blocked)')
    plt.ylabel('True Positive Rate (Caught frauds)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()