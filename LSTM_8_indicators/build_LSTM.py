import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the prepared data
X_train = np.load('LSTM_8_indicators\X_train.npy')
X_val = np.load('LSTM_8_indicators\X_val.npy')
X_test = np.load('LSTM_8_indicators\X_test.npy')
y_train = np.load('LSTM_8_indicators\y_train.npy')
y_val = np.load('LSTM_8_indicators\y_val.npy')
y_test = np.load('LSTM_8_indicators\y_test.npy')

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

#print(f"X_train shape: {X_train.shape}")
#print(f"y_train shape: {y_train.shape}")

data = pd.read_csv('data\QQQ_twelve_data_filled_30_indicators.csv')

# Use all columns in the dataset
all_columns = data.columns.tolist()  # Get all column names
all_columns = all_columns[1:]  # Redefine to exclude the first column
filtered_data = data[all_columns]

# Normalize each feature independently
scalers = {col: MinMaxScaler(feature_range=(0, 1)) for col in all_columns}
for col in all_columns:
    scalers[col].fit(filtered_data[[col]]) 

# Assuming 'close' corresponds to the target variable
target_scaler = scalers['close']

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # LSTM layers
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.dropout(x[:, -1, :])  # Use only the last output for regression
        x = self.fc(x)
        return x

input_dim = X_train.shape[2]  # Number of input features
hidden_dim = 30
output_dim = 1  # Regression output
dropout_prob = 0.2
learning_rate = 0.001
epochs = 30
batch_size = 32
step_size = 5
gamma = 0.5 


train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)


# Instantiate the model, define the loss function and optimizer
model = LSTM(input_dim, hidden_dim, output_dim, dropout_prob)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

# Training the model
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            val_predictions = model(batch_X)
            loss = criterion(val_predictions, batch_y)
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}")

# Test the model
model.eval()
with torch.no_grad():
    test_predictions = model(X_test).squeeze()
    
    # Unnormalize y_test and predictions
    y_test_original = target_scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
    test_predictions_original = target_scaler.inverse_transform(test_predictions.numpy().reshape(-1, 1))
    test_rmse = np.sqrt(mean_squared_error(y_test_original, test_predictions_original))
    print(f"Test RMSE: {test_rmse:.2f}")
    
    # Calculate Mean Absolute Error (MAE)
    test_mae = mean_absolute_error(y_test_original, test_predictions_original)
    print(f"Test MAE: {test_mae:.2f}")
    
    # Calculate R-squared (R²)
    test_r2 = r2_score(y_test_original, test_predictions_original)
    print(f"Test R²: {test_r2:.4f}")
    
    # Mean Absolute Percentage Error (MAPE)
    test_mape = np.mean(np.abs((y_test_original - test_predictions_original) / y_test_original)) * 100
    print(f"Test MAPE: {test_mape:.2f}%")
    # Plot actual vs predicted values
    plt.figure(figsize=(192, 6))
    plt.plot(y_test_original, label='Actual', color='blue')
    plt.plot(test_predictions_original, label='Predicted', color='orange')
    plt.legend(loc='lower right')
    plt.title('Actual vs Predicted Values (Original Scale)')
    plt.xlabel('Time Steps')
    plt.ylabel('Close Price')
    plt.grid()
    plt.show()

# Save the model if needed
torch.save(model.state_dict(), 'LSTM_8_indicators\lstm_model.pth')