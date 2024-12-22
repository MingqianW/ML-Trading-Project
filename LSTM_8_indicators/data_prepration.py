
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Parameters 
# #Note we get less rows for this data
lookback = 50
forward_prediction_period = 1

# Load the dataset
data = pd.read_csv('data\yfiance_data_8_indicators.csv')

# Use all columns in the dataset
all_columns = data.columns.tolist()  # Get all column names
all_columns = all_columns[1:]  # Redefine to exclude the first column
filtered_data = data[all_columns]


# Normalize each feature independently
scalers = {col: MinMaxScaler(feature_range=(0, 1)) for col in all_columns}
normalized_data = pd.DataFrame(
    {col: scalers[col].fit_transform(filtered_data[[col]])[:, 0] for col in all_columns},
    index=filtered_data.index
)

# Prepare sliding window sequences using all features
X, y = [], []
input_features = normalized_data.values

for i in range(len(input_features) - lookback - 1):
    X.append(input_features[i:i + lookback])  # Sequence of 'lookback' rows
    y.append(input_features[i + lookback, 3])  # Target: 'close' price

X, y = np.array(X), np.array(y)

# Split into training, validation, and testing sets (80:10:10 split)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)

# Reshape data for LSTM
X_train = X_train.reshape(X_train.shape[0], lookback, len(all_columns))
X_val = X_val.reshape(X_val.shape[0], lookback, len(all_columns))
X_test = X_test.reshape(X_test.shape[0], lookback, len(all_columns))

# Reshape data for LSTM
X_train = X_train.reshape(X_train.shape[0], lookback, len(all_columns))
X_val = X_val.reshape(X_val.shape[0], lookback, len(all_columns))
X_test = X_test.reshape(X_test.shape[0], lookback, len(all_columns))

# Save prepared data as .npy files
np.save('LSTM_8_indicators\X_train.npy', X_train)
np.save('LSTM_8_indicators\X_val.npy', X_val)
np.save('LSTM_8_indicators\X_test.npy', X_test)
np.save('LSTM_8_indicators\y_train.npy', y_train)
np.save('LSTM_8_indicators\y_val.npy', y_val)
np.save('LSTM_8_indicators\y_test.npy', y_test)

# Print shapes
print("Shapes of the prepared datasets:")
print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_val: {y_val.shape}")
print(f"y_test: {y_test.shape}")

# Visualization of y_test
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='y_test', color='blue')
plt.title('Visualization of y_test')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Close Price')
plt.legend()
plt.grid()
plt.show()