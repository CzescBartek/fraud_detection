import pandas as pd
from data_cleaner import financial_data_cleaner
from features import build_final_table



def load_data(data_path):
    df = pd.read_csv(data_path)
    df = financial_data_cleaner(df)
    df = build_final_table(df)