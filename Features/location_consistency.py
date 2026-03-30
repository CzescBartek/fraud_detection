import numpy as np

def get_feature_change_velocity(df):
    """Disntance between location in transactions"""
    v_cols = [f'V{i}' for i in range(1, 29)]
    v_diff = df[v_cols].diff().fillna(0)

    distance = np.sqrt((v_diff**2).sum(axis=1))
    return distance