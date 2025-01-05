import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import Tokenizer 
import utils
from config import chars
import random
from model2 import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import json
import time
import os

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
velocity_consistency_delta = 1e-5

# Define custom dataset class
class TTSDataSet(Dataset):
    def __init__(self, keys, data_path, block_size, max_timesteps):
        self.keys = keys
        self.data_path = data_path
        self.block_size = block_size
        self.max_timesteps = max_timesteps

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        src = torch.load(f"{self.data_path}/{key}_src_idx.pt", weights_only=True) #text
        tgt = torch.load(f"{self.data_path}/{key}_mel.pt", weights_only=True) #audio
        src = torch.nn.functional.pad(src, (0, self.block_size - src.size(0)), value=0)
        tgt = torch.nn.functional.pad(tgt, (0, 0, 0, self.max_timesteps - tgt.size(0)), value=-100)
        return src, tgt

# Define collate function
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch) # ts, n_mels
    src_padded = torch.stack(src_batch)
    tgt_padded = torch.stack(tgt_batch) 
    src_mask = (src_padded != 0)  # True where not padded
    audio_mask = (tgt_padded != -100).any(dim=-1) #
    return src_padded, tgt_padded, src_mask, audio_mask

# Initialize model, tokenizer, and optimizer
model = Transformer(block_size, char_size, d_model, n_heads, dropout_rate, n_layers, n_mels) 
model = model.to(device)
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print(sum(p.numel() for p in model.parameters()), 'M parameters')

# Prepare data and dataloader
data_keys = list(data_dict.keys())
random.shuffle(data_keys)
dataset = TTSDataSet(data_keys, data_path, block_size, max_timesteps)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch), pin_memory=True)
torch.set_float32_matmul_precision('high')

output_dir = "/home/kunit17/Her/Data/TrainingOutput"

# Training loop
for epoch in range(epochs):    
    epoch_loss = 0.0
    saved = False  # Ensure only one mel_output is saved per epoch
    for batch_src_idx, batch_targets, audio_mask, txt_mask in dataloader:
        t0 = time.time()
        batch_src_idx = batch_src_idx.to(device, non_blocking=True)
        x1 = batch_targets.to(device, non_blocking=True)  #target distribution
        audio_mask = audio_mask.to(device, non_blocking=True)
        txt_mask = txt_mask.to(device, non_blocking=True)
        x0 = torch.randn_like(x1) #Gaussian noise
        times = torch.rand(batch_size, device=device).view(batch_size,1,1) #(B,1,1)
        t = times * (1. - velocity_consistency_delta)
        xt = (1. - t) * x0 + t * x1 #interpolated training sample
        flow = x1 - x0
        cond = torch.where(audio_mask.unsqueeze(-1), x1, torch.zeros_like(x1))

        # Backpropagation and optimization
        optimizer.zero_grad()
        mel_output, loss = model(batch_src_idx, audio_mask, decoder_input, txt_mask, batch_targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)


        epoch_loss += loss.item()
        if not saved:
            random_idx = random.randint(0, mel_output.size(0) - 1)
            random_mel_output = mel_output[random_idx].detach().cpu()
            save_path = os.path.join(output_dir, f"mel_output_epoch_{epoch}.pth")
            torch.save(random_mel_output, save_path)
            print(f"Saved mel_output for epoch {epoch} to {save_path}")
            saved = True  # Ensure only one mel_output is saved per epoch
    print(f"step {epoch:5d} | loss: {epoch_loss:.6f} | lr | norm:  | dt: {dt:.2f}s | tok/sec: ")
        

# Save the model weights at the end of training
model_save_path = os.path.join(output_dir, "final_model_weights.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model weights saved to {model_save_path}")
