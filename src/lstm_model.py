# src/lstm_model.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ── 1. Dataset ─────────────────────────────────────────────────────────────
class ZScoreDataset(Dataset):
    """
    Converts a z-score series into sliding windows for LSTM training.
    
    Each sample is:
        X : last `seq_len` z-score values  (input sequence)
        y : next z-score value             (target to predict)
    """
    def __init__(self, zscore_series, seq_len=30):
        self.seq_len = seq_len
        self.data    = zscore_series.dropna().values.astype(np.float32)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        # LSTM expects shape (seq_len, features) — we have 1 feature (z-score)
        return torch.tensor(x).unsqueeze(-1), torch.tensor(y)


# ── 2. Model architecture ──────────────────────────────────────────────────
class LSTMPredictor(nn.Module):
    """
    A simple 2-layer LSTM that predicts the next z-score value.
    
    Architecture:
        Input  → LSTM (2 layers, 64 hidden units) → Dropout → Linear → Output
    
    input_size  : number of features per timestep (1 — just z-score)
    hidden_size : number of memory units in each LSTM layer
    num_layers  : how many LSTM layers stacked on top of each other
    dropout     : randomly zeroes some connections to prevent overfitting
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout,
            batch_first = True,   # input shape: (batch, seq_len, features)
        )
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(hidden_size, 1)   # output: single value

    def forward(self, x):
        # x shape : (batch, seq_len, 1)
        lstm_out, _ = self.lstm(x)
        # take only the last timestep's output
        last_out    = lstm_out[:, -1, :]
        out         = self.dropout(last_out)
        out         = self.linear(out)
        return out.squeeze(-1)


# ── 3. Training function ───────────────────────────────────────────────────
def train_lstm(zscore_train, seq_len=30, epochs=50, lr=0.001, batch_size=32):
    """
    Trains the LSTM on the training period z-scores.
    Returns the trained model.
    
    zscore_train : z-score Series from the train period
    seq_len      : how many past days to look at (window size)
    epochs       : how many times to pass through all training data
    lr           : learning rate — how fast the model updates its weights
    batch_size   : how many windows to process at once
    """
    dataset    = ZScoreDataset(zscore_train, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model     = LSTMPredictor()
    criterion = nn.MSELoss()              # Mean Squared Error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss        = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"  Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

    return model


# ── 4. Prediction function ─────────────────────────────────────────────────
def predict_zscore(model, zscore_full, seq_len=30):
    """
    Uses the trained LSTM to predict z-scores on the full series.
    Returns a Series of predicted z-scores aligned to the original index.
    
    For each day in the test period, we feed the model the last 30 days
    of actual z-scores and ask it to predict the next day's z-score.
    """
    import pandas as pd

    model.eval()
    data      = zscore_full.dropna().values.astype(np.float32)
    index     = zscore_full.dropna().index
    predicted = []

    with torch.no_grad():
        for i in range(seq_len, len(data)):
            x     = torch.tensor(data[i-seq_len:i]).unsqueeze(0).unsqueeze(-1)
            y_hat = model(x).item()
            predicted.append(y_hat)

    # align predictions to index — first seq_len days have no prediction
    pred_index = index[seq_len:]
    return pd.Series(predicted, index=pred_index, name="predicted_zscore")


# ── 5. Test it on one pair ─────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_prices, split_prices
    from cointegration import find_cointegrated_pairs
    from signals import get_pair_data

    prices      = load_prices()
    train, test = split_prices(prices)
    pairs       = find_cointegrated_pairs(train)

    # pick the top pair
    stock_a = pairs.iloc[0]["stock_a"]
    stock_b = pairs.iloc[0]["stock_b"]

    print(f"\nTraining LSTM on pair: {stock_a} & {stock_b}")

    # get z-score for train period
    train_data, _ = get_pair_data(train, stock_a, stock_b)
    zscore_train  = train_data["zscore"].dropna()

    # get z-score for full period (train + test) for prediction
    full_data, _  = get_pair_data(prices, stock_a, stock_b)
    zscore_full   = full_data["zscore"].dropna()

    # train
    print("\nTraining...")
    model = train_lstm(zscore_train, epochs=50)

    # predict
    print("\nPredicting z-scores...")
    predicted = predict_zscore(model, zscore_full)

    print(f"\nPredicted z-scores (last 10 days):")
    print(predicted.tail(10))