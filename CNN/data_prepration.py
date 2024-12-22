
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Parameters
lookback = 14
forward_prediction_period = 1

# Load the dataset
data = pd.read_csv('data\QQQ_twelve_data_filled_30_indicators.csv')

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

closep = normalized_data['close'].values  # Closing prices

for i in range(len(closep) - lookback - 1):
    x_i = closep[i:i + lookback]  # Closing prices for the window
    last_close = x_i[-1]  # Last closing price in the window
    next_close = closep[i + lookback]  # Future close price

    if last_close < next_close:
        y.append(1)  # Price went up
    else:
        y.append(0)  # Price went down
        
        
for i in range(len(input_features) - lookback - 1):
    X.append(input_features[i:i + lookback])  # Sequence of 'lookback' rows


X, y = np.array(X), np.array(y)
# Split into training, validation, and testing sets (80:10:10 split)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)


# Save prepared data as .npy files
np.save('CNN\X_train.npy', X_train)
np.save('CNN\X_val.npy', X_val)
np.save('CNN\X_test.npy', X_test)
np.save('CNN\y_train.npy', y_train)
np.save('CNN\y_val.npy', y_val)
np.save('CNN\y_test.npy', y_test)

# Print shapes
print("Shapes of the prepared datasets:")
print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_val: {y_val.shape}")
print(f"y_test: {y_test.shape}")

# Visualization of y_test
y_test = np.load("CNN\y_test.npy")

# Plot the distribution of classes in `y_test`
plt.figure(figsize=(12, 6))
plt.hist(y_test, bins=2, edgecolor='black', align='mid', rwidth=0.8)
plt.xticks([0, 1], labels=['Price Up', 'Price Down'])
plt.title('Distribution of Classes in y_test')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Line plot for temporal visualization
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='y_test Classes', color='blue', alpha=0.7)
plt.title('Temporal Visualization of y_test Classes')
plt.xlabel('Index')
plt.ylabel('Class')
plt.legend()
plt.grid()
plt.show()