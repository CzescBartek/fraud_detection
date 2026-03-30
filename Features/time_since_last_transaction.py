def get_time_diff(df):
    """Seconds from last transaction."""
    temp_df = df.sort_values('Time')
    time_diff = temp_df['Time'].diff().fillna(0)
    return time_diff