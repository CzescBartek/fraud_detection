from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

class Preprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.smote = SMOTE(random_state=42)
        
    
    def scale_features(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def balance_data(self, X_train, y_train):
        return self.smote.fit_resample(X_train, y_train)