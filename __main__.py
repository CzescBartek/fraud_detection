import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import joblib
import os

from Features.amount_stats import get_amount_stats
from Features.transaction_velocity import get_transaction_velocity
from Features.time_since_last_transaction import get_time_diff
from Features.location_consistency import get_feature_change_velocity
from src.data_loader import load_data
from src.model_training import FraudModelTrainer
from src.preprocessing import Preprocessor 
from src.evaluator import evaluate_model
from src.shap_analysis import run_shap_analysis
from src.model_XGB import FraudXGBoostModel
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


    rf_trainer = FraudModelTrainer()
    rf_model = rf_trainer.train(X_train_res, y_train_res)
    
    xgb_wrapper = FraudXGBoostModel(n_estimators=100, learning_rate=0.1)
    xgb_wrapper.train(X_train_res, y_train_res)


    feature_names = X_train.columns.tolist()
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(X_test_scaled, 'models/X_test.pkl') 
    joblib.dump(feature_names, 'models/feature_names.pkl') 
    joblib.dump(prep.scaler, 'models/scaler.pkl')
    joblib.dump(rf_model, 'models/rf_model.pkl')
    xgb_wrapper.save_model('models/xgb_model.pkl')


    return rf_model,xgb_wrapper.model, X_test_scaled, y_test, feature_names, prep.scaler
    

if __name__ == "__main__":
    rf_model, xgb_model, X_test, y_test, feats, scaler = main()
    
    print("\n--- EVALUATING RANDOM FOREST ---")
    evaluate_model(rf_model, X_test, y_test, feats, 'RF')
    
    print("\n--- EVALUATING XGBOOST ---")
    evaluate_model(xgb_model, X_test, y_test, feats, 'XGB')