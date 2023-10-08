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
    df['total_sales'] = df['sell_price'] * df['sales']
    
    # Convert 'date' column to datetime and extract date-related features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    #df['week_number'] = np.ceil((df['date'] - pd.Timestamp('2011-01-29')) / np.timedelta64(1, 'W')).astype(int) + 1
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Add seasonality features
    # df['season_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    # df['season_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Calculate EMA and lags
    periods = [7, 14, 21, 28]
    for period in periods:
        df[f'ema_sales_{period}'] = df['total_sales'].ewm(span=period).mean()
        #df[f'lag_sales_{period}'] = df['total_sales'].shift(period)
        df[f'rolling_std_{period}'] = df['total_sales'].rolling(window=period).std()
    
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
    
    df['total_sales'] = df['sell_price'] * df['sales']
        
    df.drop(['id', 'date', 'wm_yr_wk', 'd', 'event_name', 'event_type'], axis=1, inplace=True)
    
    return df, item_id_encoder, dept_id_encoder, store_id_encoder, state_id_encoder, cat_id_encoder

def extract_features_target(df_train, df_test, target_feature):
    X_train = df_train.drop(columns=target_feature, axis=1)
    y_train = df_train[target_feature]
    X_test = df_test.drop(columns=target_feature, axis=1)
    y_test = df_test[target_feature]
    
    return X_train, X_test, y_train, y_test

from statsmodels.tsa.stattools import adfuller
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critical Values:')
        print(f'   {key}, {value}')
    if result[1] <= 0.05:
        print("Data is stationary")
    else:
        print("Data is non-stationary")
        
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
def plot_decomposed_time_series(data, model='additive', period=365):
    seasonal_decompose_result = seasonal_decompose(data, model=model, period=period)
    
    observed = seasonal_decompose_result.observed
    trend = seasonal_decompose_result.trend
    seasonal = seasonal_decompose_result.seasonal
    residual = seasonal_decompose_result.resid
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    
    observed.plot(ax=axes[0])
    axes[0].set(ylabel='Observed')
    axes[0].set_title('Decomposition of Total Revenue')
    
    trend.plot(ax=axes[1])
    axes[1].set(ylabel='Trend')
    
    seasonal.plot(ax=axes[2])
    axes[2].set(ylabel='Seasonal')
    
    residual.plot(ax=axes[3])
    axes[3].set(ylabel='Residual')
    
    plt.tight_layout()
    plt.show()
    
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def plot_acf_pacf(data, lags=40):

    fig, (ax1, ax2) = plt.subplots(2, 1)
    plot_acf(data, lags=lags, ax=ax1)
    ax1.set_title(f'Autocorrelation Function (ACF), lags={lags}')

    plot_pacf(data, lags=lags, ax=ax2)
    ax2.set_title(f'Partial Autocorrelation Function (PACF), lags={lags}')

    plt.tight_layout()
    plt.show()