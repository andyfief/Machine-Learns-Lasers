"""
Bidirectional Temporal Convolutional Network (TCN) Demo
--------------------------------------------------------
This script builds a simple bidirectional TCN using centered (non-causal) convolutions.
It learns to predict the next sine wave value given the previous ones.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 🔧 Hyperparameters
# -----------------------------
SEQ_LEN = 30         # length of input sequence
BATCH_SIZE = 16      # number of sequences per training step
HIDDEN_CHANNELS = [16, 32, 32]  # number of filters per TCN block
KERNEL_SIZE = 3      # convolution kernel size
EPOCHS = 200         # training iterations
LR = 0.001           # learning rate


# -----------------------------
# 🧱 Bidirectional TCN Block
# -----------------------------
class BiTCNBlock(nn.Module):
    """
    One residual block of a Bidirectional TCN.
    Each block uses two Conv1D layers with centered padding (both past and future context).
    Residual connections help gradients flow through deep stacks.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # Centered padding for bidirectional lookback/lookahead
        padding = ((kernel_size - 1) * dilation) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)

        # Match input/output dims for residuals if needed
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        if self.downsample:
            x = self.downsample(x)
        return self.relu(out + x)  # residual connection


# -----------------------------
# 🧠 Bidirectional TCN Model
# -----------------------------
class BiTCN(nn.Module):
    """
    Stacks multiple BiTCNBlocks to form a multi-layer temporal model.
    Each layer’s dilation doubles, expanding receptive field exponentially.
    """
    def __init__(self, input_size, output_size, num_channels, kernel_size=3):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            dilation = 2 ** i
            layers.append(BiTCNBlock(in_ch, out_ch, kernel_size, dilation))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)  # final prediction layer

    def forward(self, x):
        y = self.network(x)
        # Take output from the last time step
        return self.fc(y[:, :, -1])


# -----------------------------
# 🧩 Data: Predict Next Sine Value
# -----------------------------
def generate_sine_batch(batch_size, seq_len):
    """Generate sine wave sequences and their next-value targets."""
    xs = np.linspace(0, 50, seq_len + 1)
    batch_x = []
    batch_y = []
    for _ in range(batch_size):
        shift = np.random.rand() * np.pi
        data = np.sin(xs + shift)
        batch_x.append(data[:-1])
        batch_y.append(data[-1])
    x = torch.tensor(batch_x, dtype=torch.float32).unsqueeze(1)  # [B, C, T]
    y = torch.tensor(batch_y, dtype=torch.float32).unsqueeze(1)  # [B, 1]
    return x, y


# -----------------------------
# 🚀 Train the Model
# -----------------------------
model = BiTCN(input_size=1, output_size=1, num_channels=HIDDEN_CHANNELS, kernel_size=KERNEL_SIZE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    x, y = generate_sine_batch(BATCH_SIZE, SEQ_LEN)
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss={loss.item():.5f}")

# -----------------------------
# 📈 Test Visualization
# -----------------------------
x_test, y_test = generate_sine_batch(1, SEQ_LEN)
with torch.no_grad():
    y_pred = model(x_test).item()

plt.plot(np.arange(SEQ_LEN), x_test.squeeze().numpy(), label="Input Sequence")
plt.scatter(SEQ_LEN, y_test.item(), color='green', label="True Next Value")
plt.scatter(SEQ_LEN, y_pred, color='red', label="Predicted Next Value")
plt.legend()
plt.title("Bidirectional TCN Sine Prediction Demo")
plt.show()
