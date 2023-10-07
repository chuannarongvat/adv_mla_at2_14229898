import pandas as pd
import numpy as np

class MeltDataFrame:
    def __init__(self, df, id_vars, value_vars_start, value_vars_end, var_name, value_name):
        self.df = df
        self.id_vars = id_vars
        self.value_vars_start = value_vars_start
        self.value_vars_end = value_vars_end
        self.var_name = var_name
        self.value_name = value_name
    
    def melt(self):
        melted_df = pd.melt(
            self.df,
            id_vars=self.id_vars,
            value_vars=[f'd_{i}' for i in range(self.value_vars_start, self.value_vars_end)],
            var_name=self.var_name,
            value_name=self.value_name
        )
        return melted_df
    
from sklearn.preprocessing import LabelEncoder    
def preprocess(df):
    # Convert 'date' column to datetime and extract date-related features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week_number'] = np.ceil((df['date'] - pd.Timestamp('2011-01-29')) / np.timedelta64(1, 'W')).astype(int) + 1
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Add seasonality features
    df['season_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['season_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Label encode categorical columns
    item_id_encoder = LabelEncoder()
    dept_id_encoder = LabelEncoder()
    store_id_encoder = LabelEncoder()
    state_id_encoder = LabelEncoder()
    cat_id_encoder = LabelEncoder()
    
    df['item_id'] = item_id_encoder.fit_transform(df['item_id'])
    df['dept_id'] = dept_id_encoder.fit_transform(df['dept_id'])
    df['store_id'] = store_id_encoder.fit_transform(df['store_id'])
    df['state_id'] = state_id_encoder.fit_transform(df['state_id'])
    df['cat_id'] = cat_id_encoder.fit_transform(df['cat_id'])
        
    df.drop(['id', 'date', 'wm_yr_wk', 'd', 'event_name', 'event_type'], axis=1, inplace=True)
    
    return df, item_id_encoder, dept_id_encoder, store_id_encoder, state_id_encoder, cat_id_encoder

def extract_features_target(df_train, df_test, target_feature):
    X_train = df_train.drop(columns=target_feature, axis=1)
    y_train = df_train[target_feature]
    X_test = df_test.drop(columns=target_feature, axis=1)
    y_test = df_test[target_feature]
    
    return X_train, X_test, y_train, y_test