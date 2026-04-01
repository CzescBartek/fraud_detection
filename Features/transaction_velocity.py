import pandas as pd

def get_transaction_velocity(df, window_seconds=3600):
    """Counting transactions in last hour """

    temp_df = df.sort_values('Time')
    
    velocity = temp_df.rolling(window=window_seconds, on='Time').count()['V1']
    return velocity