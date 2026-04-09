import xgboost as xgb
import joblib
import os

class FraudXGBoostModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42):

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss' 
        )

    def train(self, X_train, y_train):
        ratio = (y_train == 0).sum() / (y_train == 1).sum()
        self.model.set_params(scale_pos_weight=ratio)
        
        print("Training XGBoost model...")
        self.model.fit(X_train, y_train)

    def save_model(self, path='models/xgb_model.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load_model(self, path='models/xgb_model.pkl'):
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)