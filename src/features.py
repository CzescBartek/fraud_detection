from Features.amount_deviation import get_amount_deviation
from Features.transaction_velocity import get_transaction_velocity
from Features.location_consistency import get_feature_change_velocity
from Features.time_since_last_transaction import get_time_diff


def build_final_table(df):
    df['velocity'] = get_transaction_velocity(df)
    df['time_delta'] = get_time_diff(df)
    df['amt_dev'] = get_amount_deviation(df)
    df['feat_dist'] = get_feature_change_velocity(df)
    return df