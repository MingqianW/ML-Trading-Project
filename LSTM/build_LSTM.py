import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the prepared data
X_train = np.load('LSTM\X_train.npy')
X_val = np.load('LSTM\X_val.npy')
X_test = np.load('LSTM\X_test.npy')
y_train = np.load('LSTM\y_train.npy')
y_val = np.load('LSTM\y_val.npy')
y_test = np.load('LSTM\y_test.npy')

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

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


k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_results = []

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"Fold {fold + 1}/{k_folds}")

    # Split data for this fold
    train_X, val_X = X_train[train_idx], X_train[val_idx]
    train_y, val_y = y_train[train_idx], y_train[val_idx]

    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, loss, optimizer, and scheduler
    model = LSTM(input_dim, hidden_dim, output_dim, dropout_prob).to(device)  # Send model to GPU
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Train the model
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X.to(device))  # Send data to GPU
            loss = criterion(predictions, batch_y.to(device))  # Send target to GPU
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                val_predictions = model(batch_X.to(device))  # Send data to GPU
                loss = criterion(val_predictions, batch_y.to(device))  # Send target to GPU
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}")# Use normalized data here

    # Evaluate on validation fold
    val_rmse = np.sqrt(val_loss / len(val_loader))
    fold_results.append(val_rmse)
    print(f"Fold {fold + 1} RMSE: {val_rmse:.2f}")

# Aggregate cross-validation results
avg_rmse = np.mean(fold_results)
std_rmse = np.std(fold_results)
print(f"Cross-Validation Results: Average RMSE: {avg_rmse:.2f}, Std Dev RMSE: {std_rmse:.2f}") #Use normalized data here


# Test the model
    
model.eval()
with torch.no_grad():
    test_predictions = model(X_test).squeeze()  # X_test is already on GPU

    y_test_original = target_scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))
    test_predictions_original = target_scaler.inverse_transform(test_predictions.cpu().numpy().reshape(-1, 1))

    # Calculate Test RMSE
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
    
    # Plot Actual vs Predicted Values
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_original, label='Actual', color='blue')
    plt.plot(test_predictions_original, label='Predicted', color='orange')
    plt.legend(loc='lower right')
    plt.title('Actual vs Predicted Values (Original Scale)')
    plt.xlabel('Time Steps')
    plt.ylabel('Close Price')
    plt.grid()
    plt.show()

# Save the model
torch.save(model.state_dict(), 'LSTM/lstm_model.pth')

y_test_binary = (y_test_original[1:] > y_test_original[:-1]).astype(int)
predictions_binary = (test_predictions_original[1:] > test_predictions_original[:-1]).astype(int)

# Ensure y_test_binary and predictions_binary are of the same length
assert len(y_test_binary) == len(predictions_binary)

# Evaluate binary classification metrics
accuracy = accuracy_score(y_test_binary, predictions_binary)
precision = precision_score(y_test_binary, predictions_binary)
recall = recall_score(y_test_binary, predictions_binary)
f1 = f1_score(y_test_binary, predictions_binary)
conf_matrix = confusion_matrix(y_test_binary, predictions_binary)

# Print metrics
print(f"Binary Classification Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")


plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Decrease', 'Increase'], yticklabels=['Decrease', 'Increase'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Create a directory to save results
output_dir = "LSTM_result"
os.makedirs(output_dir, exist_ok=True)

# Save regression metrics to a CSV file
regression_metrics = {
    "Metric": ["RMSE", "MAE", "R²", "MAPE"],
    "Value": [test_rmse, test_mae, test_r2, test_mape]
}
regression_metrics_df = pd.DataFrame(regression_metrics)
regression_metrics_df.to_csv(os.path.join(output_dir, "regression_metrics.csv"), index=False)

# Save binary classification metrics to a CSV file
classification_metrics = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [accuracy, precision, recall, f1]
}
classification_metrics_df = pd.DataFrame(classification_metrics)
classification_metrics_df.to_csv(os.path.join(output_dir, "classification_metrics.csv"), index=False)

# Save the confusion matrix as an image
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Decrease', 'Increase'], yticklabels=['Decrease', 'Increase'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(confusion_matrix_path)
plt.close()

# Save the Actual vs Predicted plot
plt.figure(figsize=(12, 6))
plt.plot(y_test_original, label='Actual', color='blue')
plt.plot(test_predictions_original, label='Predicted', color='orange')
plt.legend(loc='lower right')
plt.title('Actual vs Predicted Values (Original Scale)')
plt.xlabel('Time Steps')
plt.ylabel('Close Price')
plt.grid()
actual_vs_predicted_path = os.path.join(output_dir, "actual_vs_predicted.png")
plt.savefig(actual_vs_predicted_path)
plt.close()

print(f"Results saved in '{output_dir}' directory.")