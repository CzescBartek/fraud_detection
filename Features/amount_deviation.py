def get_amount_deviation(df, window=10):
    """Checking how price is deviating from last transactions"""
    rolling_mean = df['scaled_amount'].rolling(window=window).mean()


    deviation = df['scaled_amount'] / (rolling_mean + 1e-9) 
    return deviation.fillna(1)