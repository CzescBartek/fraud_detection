import numpy as np

def get_feature_change_velocity(df):
    """Disntance between location in transactions"""

    v_diff = df[['V1', 'V2', 'V3']].diff().fillna(0)

    distance = np.sqrt((v_diff**2).sum(axis=1))
    return distance