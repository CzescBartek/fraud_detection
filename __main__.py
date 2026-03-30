import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import joblib
import os

from Features.amount_deviation import get_amount_deviation
from Features.transaction_velocity import get_transaction_velocity
from Features.time_since_last_transaction import get_time_diff
from Features.location_consistency import get_feature_change_velocity
from src.data_loader import load_data
from src.model_training import FraudModelTrainer
from src.preprocessing import Preprocessor 
from src.evaluator import evaluate_model
from src.shap_analysis import run_shap_analysis
def main():

    df = load_data('data/creditcard.csv')
    df = df.dropna()
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


    prep = Preprocessor()
    

    X_train_scaled, X_test_scaled = prep.scale_features(X_train, X_test)
    X_train_res, y_train_res = prep.balance_data(X_train_scaled, y_train)


    trainer = FraudModelTrainer()
    model = trainer.train(X_train_res, y_train_res)
    
    feature_names = X_train.columns.tolist()
    
    print("\nPROCES FINISHED")
    print(f"AMOUNT OF FEATURES IN MODEL:  {len(feature_names)}")
    print("Zapisywanie danych do analizy SHAP...")
    joblib.dump(X_test_scaled, 'models/X_test.pkl') 
    joblib.dump(feature_names, 'models/feature_names.pkl') 
    return model, X_test_scaled, y_test, feature_names
    

if __name__ == "__main__":
    model, X_test, y_test, feats = main()
    evaluate_model(model, X_test, y_test, feats)