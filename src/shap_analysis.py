import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_shap_analysis(model_path, X_test_path, feature_names_path):
    print("\n--- SHAP analysis ---")
    
    # 1. Wczytywanie modelu i danych
    if not os.path.exists(model_path):
        print(f"No model : {model_path}")
        return
        
    model = joblib.load(model_path)
    X_test = joblib.load(X_test_path) 
    feature_names = joblib.load(feature_names_path) 


    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    print("Sampling 500 rows")
    X_sample = shap.sample(X_test_df, 500)

    explainer = shap.TreeExplainer(model)
    

    print("Computing shap. May take a while.")

    all_shap_values = explainer.shap_values(X_sample)
    if isinstance(all_shap_values, list):
        shap_values_to_plot = all_shap_values[1]
    else:
    # Jeśli to tablica 3D, bierzemy ostatni wymiar odpowiadający klasie 1
        shap_values_to_plot = all_shap_values[:, :, 1]

    print("Generowanie wykresu Summary Plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_to_plot, X_sample, show=False)
    plt.show()


    example_index = 0

    explanation = shap.Explanation(
        values=shap_values[example_index],
        base_values=explainer.expected_value[1],
        data=X_sample.iloc[example_index],
        feature_names=feature_names
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.title(f"SHAP Waterfall Plot: Analiza transakcji nr {X_sample.index[example_index]}")
    plt.tight_layout()
    plt.savefig('models/shap_waterfall.png')
    plt.show()

    print("--- Analiza SHAP zakończona ---")

if __name__ == "__main__":

    run_shap_analysis(
        '../models/fraud_model.pkl', 
        '../models/X_test.pkl', 
        '../models/feature_names.pkl'
    )