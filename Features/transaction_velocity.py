import pandas as pd

def get_transaction_velocity(df, window_seconds=3600):
    """Counting transactions in last hour """

    temp_df = df.sort_values('scaled_time')
    
    velocity = temp_df.rolling(window=f'{window_seconds}s', on='scaled_time').count()['V1']
    return velocity