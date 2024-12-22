import requests
import pandas as pd
import time

# Your Twelve Data API key
api_key = 'd8f9d38fc83741639c006290903d21a3'

# The stock symbol you want to query (QQQ in this case)
stock_symbol = "QQQ"

# Initialize an empty DataFrame for 'df'
df = pd.DataFrame()

# Base URLs for Twelve Data API
base_url = "https://api.twelvedata.com/time_series"
adx_url = "https://api.twelvedata.com/adx"
atr_url = "https://api.twelvedata.com/atr"
wma_url = "https://api.twelvedata.com/wma"
cci_url = "https://api.twelvedata.com/cci"
roc_url = "https://api.twelvedata.com/roc"
stoch_url = "https://api.twelvedata.com/stoch"
sar_url = "https://api.twelvedata.com/sar"
mfi_url = "https://api.twelvedata.com/mfi"
vwap_url = "https://api.twelvedata.com/vwap"

# Wait time in seconds between API requests (to avoid exceeding the rate limit)
wait_time = 8

# Fetch and merge function with debugging and empty DataFrame handling
def fetch_and_merge(url, params, columns):
    global df
    response = requests.get(url, params=params)
    
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        print(f"Failed to decode JSON from {url}. Response text: {response.text}")
        return df
    
    if "values" in data:
        ind_df = pd.DataFrame(data["values"])
        ind_df['datetime'] = pd.to_datetime(ind_df['datetime'])
        
        # Print out the columns to inspect the response structure
        print("Columns in API response:", ind_df.columns)
        
        # Check if all required columns are present
        missing_columns = [col for col in columns if col not in ind_df.columns]
        if missing_columns:
            print(f"Warning: The following columns are missing from the API response: {missing_columns}")
            # Set the missing columns to NaN if not present
            for col in missing_columns:
                ind_df[col] = float('nan')
        
        # Merge with existing DataFrame
        if df.empty:
            df = ind_df[['datetime'] + columns]
        else:
            df = df.merge(ind_df[['datetime'] + columns], on='datetime', how='left')
        
        return df
    else:
        print(f"Failed to fetch data from {url}")
        print(data)  # Print the full response for debugging
        return df

# Fetch time series data (open, high, low, close, volume)
params = {
    "symbol": stock_symbol,
    "interval": "1day",  # You can change this to '1min', '15min', etc.
    "outputsize": "5000",  # Fetch up to 5000 data points
    "apikey": api_key
}
response = requests.get(base_url, params=params)

# Convert the response to JSON
data = response.json()

if "values" in data:
    # Convert the JSON data to a Pandas DataFrame
    df_price = pd.DataFrame(data["values"])
    
    # Convert 'datetime' to a pandas datetime object and other columns to numeric
    df_price['datetime'] = pd.to_datetime(df_price['datetime'])
    df_price[['open', 'high', 'low', 'close', 'volume']] = df_price[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
    
    # Sort the DataFrame by date
    df_price = df_price.sort_values(by="datetime")
    
    # Merge price data with existing df
    if df.empty:
        df = df_price[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    else:
        df = df.merge(df_price[['datetime', 'open', 'high', 'low', 'close', 'volume']], on='datetime', how='left')

time.sleep(wait_time)  # Wait between API requests

# Fetch ADX
adx_params = {"symbol": stock_symbol, "interval": "1day", "apikey": api_key}
df = fetch_and_merge(adx_url, adx_params, ['adx', 'plus_di', 'minus_di'])
time.sleep(wait_time)

# Fetch ATR
atr_params = {"symbol": stock_symbol, "interval": "1day", "apikey": api_key}
df = fetch_and_merge(atr_url, atr_params, ['atr'])
time.sleep(wait_time)

# Fetch WMA, CCI, ROC, Stochastic, Williams %R, SAR, MFI, VWAP
wma_params = {"symbol": stock_symbol, "interval": "1day", "time_period": 50, "series_type": "close", "apikey": api_key}
df = fetch_and_merge(wma_url, wma_params, ['wma'])
time.sleep(wait_time)

cci_params = {"symbol": stock_symbol, "interval": "1day", "apikey": api_key}
df = fetch_and_merge(cci_url, cci_params, ['cci'])
time.sleep(wait_time)

roc_params = {"symbol": stock_symbol, "interval": "1day", "apikey": api_key}
df = fetch_and_merge(roc_url, roc_params, ['roc'])
time.sleep(wait_time)

stoch_params = {"symbol": stock_symbol, "interval": "1day", "apikey": api_key}
df = fetch_and_merge(stoch_url, stoch_params, ['slow_k', 'slow_d'])
time.sleep(wait_time)

sar_params = {"symbol": stock_symbol, "interval": "1day", "apikey": api_key}
df = fetch_and_merge(sar_url, sar_params, ['sar'])
time.sleep(wait_time)

mfi_params = {"symbol": stock_symbol, "interval": "1day", "apikey": api_key}
df = fetch_and_merge(mfi_url, mfi_params, ['mfi'])
time.sleep(wait_time)

vwap_params = {"symbol": stock_symbol, "interval": "1day", "apikey": api_key}
df = fetch_and_merge(vwap_url, vwap_params, ['vwap'])
time.sleep(wait_time)


# Save the final DataFrame with stock prices and additional indicators
df.to_csv(f'data\{stock_symbol}_twelve_data_with_all_indicators_and_prices.csv', index=False)

print(f"Data for {stock_symbol} collected successfully with stock prices and all indicators, saved to {stock_symbol}_twelve_data_with_all_indicators_and_prices.csv!")
