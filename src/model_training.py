from sklearn.ensemble import RandomForestClassifier
import joblib

class FraudModelTrainer:
    def __init__(self):

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,  
            verbose=1
        )

    def train(self, X_train, y_train):
        print("Start training Random Forest...")
        self.model.fit(X_train, y_train)

        joblib.dump(self.model, 'models/fraud_model.pkl')
        return self.model

    def predict_probs(self, X_test):
  
        return self.model.predict_proba(X_test)[:, 1]