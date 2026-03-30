import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
import pandas as pd

def evaluate_model(model, X_test, y_test, feature_names):

    y_probs = model.predict_proba(X_test)[:, 1]
    
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importances, palette='viridis')
    plt.title('Top 10 najważniejszych cech (w tym Twoje autorskie!)')
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='b', label=f'AP = {average_precision_score(y_test, y_probs):.2f}')
    plt.xlabel('Recall (How many frauds)')
    plt.ylabel('Precision (Client wrongly blocked)')
    plt.title('Precision-Recall curve')
    plt.legend()
    plt.show()