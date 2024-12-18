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
import time

# Load data_dict from the JSON file
with open("data_dict.json", "r") as f:
    data_dict = json.load(f)

# Set manual seed and device
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load training parameters
batch_size, learning_rate, epochs, block_size, char_size, d_model, n_heads, dropout_rate, head_size = utils.get_train_params() #take out head_size
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
    src_batch, tgt_batch = zip(*batch) # ts, n_mels
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0) # will make all text input same max length
    tgt_padded = torch.stack(tgt_batch) 
    src_mask = (src_padded != 0)  # True where not padded
    tgt_mask = (tgt_padded != -100).any(dim=-1)
    return src_padded, tgt_padded, src_mask, tgt_mask

# Initialize model, tokenizer, and optimizer
model = Transformer(block_size, char_size, d_model, n_heads, dropout_rate, n_layers, n_mels) 
model = model.to(device)
#model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print(sum(p.numel() for p in model.parameters()), 'M parameters')

# Prepare data and dataloader
data_keys = list(data_dict.keys())
random.shuffle(data_keys)
dataset = TTSDataSet(data_keys, data_path, max_timesteps)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch), pin_memory=True)
torch.set_float32_matmul_precision('high')

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    t0 = time.time()
    epoch_loss = 0
    for batch_src_idx, batch_targets, encoder_mask, decoder_mask in dataloader:
        batch_src_idx = batch_src_idx.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)
        encoder_mask = encoder_mask.to(device, non_blocking=True)
        print(f' decoder mask: {decoder_mask.shape}, decoder input {batch_targets.shape}')
        decoder_mask = decoder_mask.to(device, non_blocking=True)        
        decoder_input = batch_targets.clone() # (ts, n_mels) # Create a copy of the targets for the decoder input     
        print(decoder_input.dtype)
        # Backpropagation and optimization
        optimizer.zero_grad()
        mel_output, loss = model(batch_src_idx, encoder_mask, decoder_input, decoder_mask, batch_targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)*1000
        tokens_sec = (batch_src_idx.shape[0] * batch_src_idx.shape[1]) / (t1-t0)
        epoch_loss += loss.item()
        print(f"step {epoch:5d} | loss: {epoch_loss.item():.6f} | lr | norm:  | dt: {dt*1000:.2f}ms | tok/sec: {tokens_sec:.2f}")
        break

    print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader)}")

