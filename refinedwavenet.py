
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Load Data
train_df = pd.read_csv('/train.csv')
test_df = pd.read_csv('/test.csv')
labels_df = pd.read_csv('/test_label.csv')

# Normalize using training statistics
mean, std = train_df['TOTALDEMAND'].mean(), train_df['TOTALDEMAND'].std()
train_df['TOTALDEMAND'] = (train_df['TOTALDEMAND'] - mean) / std
test_df.iloc[:, 1] = (test_df.iloc[:, 1] - mean) / std


train_data = train_df['TOTALDEMAND'].values.astype(np.float32)
test_data = test_df.iloc[:, 1].values.astype(np.float32)
anomaly_labels = labels_df.iloc[:, 1].values.astype(int)  # Assuming first column contains labels


# Define Dataset Class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=55):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):
        x = self.data[index : index + self.seq_len]
        y = self.data[index + self.seq_len]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(y, dtype=torch.float32).unsqueeze(0)
# Hyperparameters
SEQ_LEN = 55
HIDDEN_DIM = 64
NUM_LAYERS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 30

# Create DataLoaders
split = int(len(train_data) * 0.8)
train_dataset = TimeSeriesDataset(train_data[:split], seq_len=SEQ_LEN)
val_dataset = TimeSeriesDataset(train_data[split:], seq_len=SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedWaveNetWithRefiner(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, num_layers=10,
                 kernel_size=2, dilation_base=2, dropout=0.2,
                 refiner_type='gru'):  # 'gru' or 'cnn'
        super().__init__()
        self.refiner_type = refiner_type.lower()
        assert self.refiner_type in ['gru', 'cnn'], "refiner_type must be 'gru' or 'cnn'"

        self.res_blocks = nn.ModuleList()
        self.gate_blocks = nn.ModuleList()
        self.skip_blocks = nn.ModuleList()
        self.dropout_blocks = nn.ModuleList()

        for i in range(num_layers):
            dilation = dilation_base ** i
            padding = (kernel_size - 1) * dilation
            in_ch = in_channels if i == 0 else hidden_dim

            self.res_blocks.append(nn.Conv1d(in_ch, hidden_dim, kernel_size, padding=padding, dilation=dilation))
            self.gate_blocks.append(nn.Conv1d(in_ch, hidden_dim, kernel_size, padding=padding, dilation=dilation))
            self.skip_blocks.append(nn.Conv1d(hidden_dim, hidden_dim, 1))
            self.dropout_blocks.append(nn.Dropout(p=dropout))

        if self.refiner_type == 'gru':
            self.refiner = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim,
                                  num_layers=1, batch_first=True)
        elif self.refiner_type == 'cnn':
            self.refiner = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x shape: (B, C=1, T)
        skip_sum = None
        for conv_f, conv_g, skip, drop in zip(self.res_blocks, self.gate_blocks, self.skip_blocks, self.dropout_blocks):
            f = torch.tanh(conv_f(x))
            g = torch.sigmoid(conv_g(x))
            out = drop(f * g)[..., :x.shape[-1]]
            skip_out = skip(out)
            skip_sum = skip_out if skip_sum is None else skip_sum + skip_out
            x = x + out if out.shape == x.shape else out

        if self.refiner_type == 'gru':
            refined_input = skip_sum.permute(0, 2, 1)  # (B, T, C)
            _, h = self.refiner(refined_input)
            out = h[-1]  # Last GRU hidden state
        else:  # CNN refiner
            out = self.refiner(skip_sum)  # (B, C, T)
            out = torch.mean(out, dim=2)  # Global average pooling

        return self.output_layer(out).squeeze(1)  # (B

model = GatedWaveNetWithRefiner(
    in_channels=1,
    hidden_dim=64,
    num_layers=10,
    refiner_type='gru',  # or 'cnn'
).to(device)
criterion = nn.SmoothL1Loss(beta=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

best_val_loss = float("inf")
patience = 5
counter = 0

# Training
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            val_loss += criterion(predictions, batch_y).item()
    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("⏹️ Early stopping triggered.")
            break

print("✅ Training Complete!")

SEQ_LEN=48

# Prepare Test Dataset
SEQ_LEN=48
test_dataset = TimeSeriesDataset(test_data, seq_len=SEQ_LEN)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Make Predictions
model.eval()
predictions = []
actual_values = test_data[SEQ_LEN:]
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        pred = model(batch_x).cpu().item()
        predictions.append(pred)

# Convert to NumPy Arrays
predictions = np.array(predictions)
print(len(predictions))
actual_values = np.array(actual_values)
print(len(actual_values))
errors = np.abs(predictions - actual_values)

print("✅ Predictions aligned with actual values. Errors computed correctly!")

true_labels = labels_df.iloc[:, 1].values.astype(int)[SEQ_LEN:]

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# Compute 90th percentile as the anomaly detection threshold
best_threshold = np.percentile(errors, 90)
print(f"✅ 90th Percentile Threshold Found: {best_threshold:.4f}")

# Anomaly Detection
anomalies = np.where(errors > best_threshold)[0]
print(f"📌 Detected Anomalies at indices: {anomalies}")

df_anomalies = pd.DataFrame(anomalies, columns=['Anomaly_Index'])
df_anomalies.to_csv("anomalies_indicesWavenet.csv", index=False)
print("📁 Anomaly indices saved to 'anomalies_indicesWavenet.csv'")

# Generate Predictions
pred_labels = np.zeros_like(true_labels)
pred_labels[anomalies] = 1

# Compute Metrics
precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)
roc_auc = roc_auc_score(true_labels, errors)  # Using errors as the score
pr_auc = average_precision_score(true_labels, errors)  # Using errors as the score

print(f"📊 Precision: {precision:.4f}")
print(f"📊 Recall: {recall:.4f}")
print(f"📊 F1 Score: {f1:.4f}")
print(f"📊 ROC AUC: {roc_auc:.4f}")
print(f"📊 PR AUC: {pr_auc:.4f}")
