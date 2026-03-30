def financial_data_cleaner(df):
    ''' 
        Deleting noise in our data
    '''
    df = df[df['Amount'] > 0]
    

    df = df.drop_duplicates(subset=['Time', 'Amount', 'V1', 'V2'], keep='first')
    
    # Setting high value column to get outliers
    q99 = df['Amount'].quantile(0.99)
    df['is_high_value'] = (df['Amount'] > q99).astype(int)
    
    return df