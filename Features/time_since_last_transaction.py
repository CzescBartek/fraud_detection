def get_time_diff(df):
    """Seconds from last transaction."""
    temp_df = df.sort_values('scaled_time')
    time_diff = temp_df['scaled_time'].diff().fillna(0)
    return time_diff