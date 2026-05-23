import pandas as pd
import numpy as np
import os

def process_data(data_dir, output_file, sample_size=100000):
    print("Loading datasets (with sampling for performance)...")
    
    train_path = os.path.join(data_dir, 'train.csv')
    df_train = pd.read_csv(train_path)
    
    members_path = os.path.join(data_dir, 'members_v3.csv')
    df_members = pd.read_csv(members_path)
    
    transactions_path = os.path.join(data_dir, 'transactions.csv')
    df_transactions = pd.read_csv(transactions_path, nrows=sample_size)
    
    user_logs_path = os.path.join(data_dir, 'user_logs.csv')
    df_logs = pd.read_csv(user_logs_path, nrows=sample_size)
    
    print("Engineering Advanced Features...")
    
    # 1. Customer Lifetime Value (CLV) & Payment Behavior
    trans_features = df_transactions.groupby('msno').agg({
        'actual_amount_paid': 'sum', # CLV
        'is_cancel': 'sum',          # Total Cancellations
        'is_auto_renew': 'last',     # Current Auto-Renew Status
        'payment_plan_days': 'mean'  # Average plan duration
    }).reset_index()
    trans_features.rename(columns={'actual_amount_paid': 'CLV', 'is_cancel': 'Total_Cancellations'}, inplace=True)
    
    # 2. Usage Trends & Engagement
    # Convert date to datetime to sort
    df_logs['date'] = pd.to_datetime(df_logs['date'], format='%Y%m%d')
    df_logs = df_logs.sort_values(by=['msno', 'date'])
    
    # We take the total seconds listened and the number of unique songs
    logs_features = df_logs.groupby('msno').agg({
        'total_secs': 'mean',        # Avg listening time per day
        'num_unq': 'mean',           # Avg unique songs per day
    }).reset_index()
    logs_features.rename(columns={'total_secs': 'Avg_Daily_Listening_Secs', 'num_unq': 'Avg_Daily_Unique_Songs'}, inplace=True)
    
    print("Joining tables...")
    
    # Merge all into train
    df = df_train.merge(df_members, on='msno', how='left')
    df = df.merge(trans_features, on='msno', how='left')
    df = df.merge(logs_features, on='msno', how='left')
    
    # Clean up and prepare for ML
    df['city'] = df['city'].fillna(-1)
    df['registered_via'] = df['registered_via'].fillna(-1)
    
    # Drop IDs and non-numeric dates for training
    drop_cols = ['msno', 'gender', 'registration_init_time', 'bd']
    df_numeric = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    
    # Fill remaining NAs with 0
    df_numeric = df_numeric.fillna(0)
    
    print(f"Saving finalized data to {output_file}...")
    df_numeric.to_csv(output_file, index=False)
    print("Preprocessing complete!")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, '../data'))
    output_file = os.path.abspath(os.path.join(current_dir, '../data/processed_train.csv'))
    
    process_data(data_dir, output_file)
