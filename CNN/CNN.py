import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

# Load the prepared data
X_train = np.load('CNN/X_train.npy')
X_val = np.load('CNN/X_val.npy')
X_test = np.load('CNN/X_test.npy')
y_train = np.load('CNN/y_train.npy')
y_val = np.load('CNN/y_val.npy')
y_test = np.load('CNN/y_test.npy')

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
lookback = X_train.shape[1]
X_train = X_train.reshape(X_train.shape[0], lookback, 5, 6) 
X_val = X_val.reshape(X_val.shape[0], lookback, 5, 6)
X_test = X_test.reshape(X_test.shape[0], lookback, 5, 6)
print("Shapes of the prepared datasets in 2D:")
print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_val: {y_val.shape}")
print(f"y_test: {y_test.shape}")

# Dataset and DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        out = self.bn2(self.conv2(out))
        out += identity
        return F.leaky_relu(out, negative_slope=0.01)

class CNNModel(nn.Module):
    def __init__(self, lookback = 14, output_classes=2, dropout_prob=0.5):
        super(CNNModel, self).__init__()
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(in_channels=lookback, out_channels=64, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.res1 = ResidualBlock(64, 64)
        self.res2 = ResidualBlock(64, 128, stride=2, downsample=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=2),
            nn.BatchNorm2d(128),
        ))
        self.res3 = ResidualBlock(128, 256, stride=2, downsample=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 1), stride=2),
            nn.BatchNorm2d(256),
        ))
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_classes)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


model = CNNModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-8)

num_epochs = 30
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()  
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        
        optimizer.step()
        train_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))
    print(f"Epoch {epoch+1}, Loss: {train_loss/len(train_loader):.4f}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() 

    # Average validation loss over all batches
    val_loss /= len(val_loader)

    # Append to val_losses
    val_losses.append(val_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # Step the scheduler
    scheduler.step(val_loss)

# Evaluate the model on the test set
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)  # Get log probabilities
        predictions = torch.argmax(outputs, dim=1)  # Convert to class predictions
        y_true.extend(y_batch.numpy())
        y_pred.extend(predictions.numpy())

# Convert to NumPy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate classification metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_true, y_pred)

# Print classification report
print("Classification Report:\n", classification_report(y_true, y_pred))

# Print confusion matrix
print("Confusion Matrix:\n", conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Price went down', 'Price went up'], yticklabels=['Price went down', 'Price went up'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save metrics and plots
output_dir = "CNN_results"
os.makedirs(output_dir, exist_ok=True)

# Save classification metrics
classification_metrics = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [accuracy, precision, recall, f1]
}
classification_metrics_df = pd.DataFrame(classification_metrics)
classification_metrics_df.to_csv(os.path.join(output_dir, "classification_metrics.csv"), index=False)

# Save confusion matrix plot
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Price went down', 'Price went up'], yticklabels=['Price went down', 'Price went up'])
plt.title("Confusion Matrix")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(confusion_matrix_path)
plt.close()

print(f"Results saved in '{output_dir}' directory.")
