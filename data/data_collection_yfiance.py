import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pandas_ta as ta

# Define the stock code and the date range
stock_code = "QQQ"
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')  # Last 10 years

# Download the stock data using yfinance
data = yf.download(stock_code, start=start_date, end=end_date)
data.to_csv("raw_last_10_years.csv")

# Load the data, skipping the first two rows
data = pd.read_csv('raw_last_10_years.csv', skiprows=2)

# Ensure proper column names
data.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

# Convert necessary columns to numeric values
cols_to_convert = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
data[cols_to_convert] = data[cols_to_convert].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
data.dropna(inplace=True)

# Now you can proceed to calculate MACD, ADX, and other indicators
macd = ta.macd(data['Close'])
if macd is not None and not macd.empty:
    data['MACD'] = macd.iloc[:, 0].fillna(0)  # Fill missing MACD values with 0
else:
    data['MACD'] = 0  # Use 0 if MACD calculation fails

adx = ta.adx(data['High'], data['Low'], data['Close'])
if adx is not None and not adx.empty:
    data['ADX'] = adx.iloc[:, 0].fillna(0)  # Fill missing ADX values with 0
else:
    data['ADX'] = 0  # Use 0 if ADX calculation fails

# Save the cleaned and processed data
data.to_csv('data\yfiance_data_8_indicators.csv', index=False)
