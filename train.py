import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import Tokenizer 
import utils
from config import chars
import random
from model import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import json

# Load data_dict from the JSON file
with open("data_dict.json", "r") as f:
    data_dict = json.load(f)

# Set manual seed and device
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load training parameters
batch_size, learning_rate, epochs, block_size, char_size, d_model, n_heads, dropout_rate, head_size = utils.get_train_params()
n_layers = 5
n_mels = 128
max_timesteps = 302
data_path = utils.get_training_data()

# Define custom dataset class
class TTSDataSet(Dataset):
    def __init__(self, keys, data_path, max_timesteps):
        self.keys = keys
        self.data_path = data_path
        self.max_timesteps = max_timesteps

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        src = torch.load(f"{self.data_path}/{key}_src_idx.pt")
        tgt = torch.load(f"{self.data_path}/{key}_mel.pt")
        tgt = torch.nn.functional.pad(tgt, (0, 0, 0, self.max_timesteps - tgt.size(0)), value=-100)
        return src, tgt

# Define collate function
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=-100)
    src_mask = (src_padded != 0)  # True where not padded
    tgt_mask = (tgt_padded != -100)
    return src_padded, tgt_padded, src_mask, tgt_mask

# Initialize model, tokenizer, and optimizer
model = Transformer(block_size, char_size, d_model, n_heads, dropout_rate, head_size, n_layers, n_mels, device) 
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()  # Example loss function, adjust as needed
print(sum(p.numel() for p in model.parameters()), 'M parameters')

# Prepare data and dataloader
data_keys = list(data_dict.keys())
print(data_keys)
random.shuffle(data_keys)
dataset = TTSDataSet(data_keys, data_path, max_timesteps)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch), pin_memory=True)

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    model.train()
    epoch_loss = 0
    for batch_src_idx, batch_targets, encoder_mask, decoder_mask in dataloader:
        # Create a copy of the targets for the decoder input
        # Move data to GPU
        batch_src_idx = batch_src_idx.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        encoder_mask = encoder_mask.to(device, non_blocking=True)
        decoder_mask = decoder_mask.to(device, non_blocking=True)

        # Create a copy of the targets for the decoder input
        decoder_input = batch_targets.clone()
        
        # Forward pass
        pred = model(batch_src_idx, encoder_mask)  # Replace with actual model forward logic

        # Compute loss
        loss = criterion(pred, batch_targets)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()
        break

    print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader)}")

