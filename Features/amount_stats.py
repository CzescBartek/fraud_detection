def get_amount_stats(df, window=10):
    """
    Checking how price is deviating using both Ratio and Z-Score
    """
    rolling_mean = df['Amount'].rolling(window=window).mean()
    rolling_std = df['Amount'].rolling(window=window).std()
    
    # Twój stary Ratio
    ratio = df['Amount'] / (rolling_mean + 1e-9)
    
    # Nowy Z-Score (bardziej odporny na szum)
    zscore = (df['Amount'] - rolling_mean) / (rolling_std + 1e-9)
    
    return ratio.fillna(1), zscore.fillna(0)