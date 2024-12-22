import pandas as pd
import pandas_ta as ta

def fill_missing_indicators(df):
    # Convert datetime column to proper datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Sort the DataFrame by datetime
    df = df.sort_values('datetime')
    
    # Set the datetime as the index (this helps with VWAP calculation)
    df.set_index('datetime', inplace=True)

    # Calculate the missing indicators using Pandas_TA
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx['ADX_14'].fillna(0)
    df['plus_di'] = adx['DMP_14'].fillna(0)
    df['minus_di'] = adx['DMN_14'].fillna(0)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14).fillna(0)
    df['wma'] = ta.wma(df['close'], length=14).fillna(0)
    df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20).fillna(0)
    df['roc'] = ta.roc(df['close'], length=14).fillna(0)
    
    stoch = ta.stoch(df['high'], df['low'], df['close'])
    df['slow_k'] = stoch['STOCHk_14_3_3'].fillna(0)
    df['slow_d'] = stoch['STOCHd_14_3_3'].fillna(0)

    psar = ta.psar(df['high'], df['low'], df['close'])
    df['sar'] = psar['PSARl_0.02_0.2'].fillna(0)

    df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14).fillna(0)
    
    # Calculate VWAP (requires sorted datetime index)
    df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume']).fillna(0)

    # Add additional indicators
    df['sma_5'] = ta.sma(df['close'], length=5).fillna(0)
    df['sma_10'] = ta.sma(df['close'], length=10).fillna(0)
    df['sma_20'] = ta.sma(df['close'], length=20).fillna(0)
    df['ema_9'] = ta.ema(df['close'], length=9).fillna(0)
    df['ema_20'] = ta.ema(df['close'], length=20).fillna(0)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9).fillna(0)
    df['macd'] = macd['MACD_12_26_9'].fillna(0)
    df['macd_signal'] = macd['MACDs_12_26_9'].fillna(0)
    df['macd_hist'] = macd['MACDh_12_26_9'].fillna(0)
    bb = ta.bbands(df['close'], length=20, std=2)
    df['bb_upper'] = bb['BBU_20_2.0'].fillna(0)
    df['bb_middle'] = bb['BBM_20_2.0'].fillna(0)
    df['bb_lower'] = bb['BBL_20_2.0'].fillna(0)
    df['rsi'] = ta.rsi(df['close'], length=14).fillna(0)
    df['obv'] = ta.obv(df['close'], df['volume']).fillna(0)

    # Remove any remaining NaNs after adding indicators
    df.dropna(inplace=True)

    return df

# Load the CSV file
df = pd.read_csv("data\QQQ_twelve_data_with_all_indicators_and_prices.csv")

# Fill missing indicators
df_filled = fill_missing_indicators(df)

# Save the filled data to a new CSV
df_filled.to_csv('data\QQQ_twelve_data_filled_30_indicators.csv')
