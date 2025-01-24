import torch
import torch.nn as nn
from torch.nn import functional as F
import utils
from config import chars
from model2 import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import json
import time
import os
from torch.utils.tensorboard import SummaryWriter
from utils import Tokenizer  # Assuming your Tokenizer class is in utils.py

# Load data_dict from the JSON file
with open("shart_dict.json", "r") as f:
    data_dict = json.load(f)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load training parameters
batch_size, learning_rate, epochs, char_size, d_model, n_heads, dropout_rate, head_size, n_layers = utils.get_train_params() #take out head_size
n_fft, hop_length, sr, n_mels, max_timesteps = utils.get_audio_params()
training_mels, training_text = utils.get_training_data()

# Define custom dataset class
class TTSDataSet(Dataset):
    def __init__(self, keys, training_mels, training_text, max_timesteps):
        self.keys = keys
        self.mels_data_path = training_mels
        self.text_data_path = training_text
        self.max_timesteps = max_timesteps

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):  
        key = self.keys[idx]
        txt = torch.load(f"{self.text_data_path}/{key}.pt", weights_only=True) #text S
        tgt = torch.load(f"{self.mels_data_path}/{key}.pt", weights_only=True) #audio S,C
        txt = torch.nn.functional.pad(txt, (0, tgt.size(0) - txt.size(0)), value=4)  #this adds special filler tokens "FILL" to align txt text (S,) dimension with tgt audio dimension (S,)
        txt = torch.nn.functional.pad(txt, (0, self.max_timesteps - txt.size(0)), value=0) #adds padding to max timesteps 
        tgt = torch.nn.functional.pad(tgt, (0, 0, 0, self.max_timesteps - tgt.size(0)), value=-3) # s,c where padded s gets -3 and c relating to the rows get -3
        return txt, tgt

# Define collate function
def collate_fn(batch):
    txt_batch, tgt_batch = zip(*batch)  
    txt_padded = torch.stack(txt_batch) # txt -> (B, S), 
    tgt_padded = torch.stack(tgt_batch) # tgt -> (B, S, C)
    attn_mask = (tgt_padded != -3).any(dim=-1)  # Shape: (B, max_timesteps) - attn mask has true for non-padded positions
    valid_lengths = attn_mask.sum(dim=-1)  # Shape: (B,), number of valid timesteps per sequence
    # Apply 70-100% contiguous masking only to the valid portion
    random_mask = torch.zeros_like(attn_mask, dtype=torch.bool)  # Initialize with all False
    for i, valid_len in enumerate(valid_lengths):
        mask_percentage = torch.empty(1).uniform_(0.7, 1.0).item()  # Random percentage between 70% and 100%
        timesteps_to_mask = int(mask_percentage * valid_len)  # Compute timesteps to mask
        temp_mask = torch.zeros(valid_len, dtype=torch.bool)             # Generate a contiguous mask for valid timesteps
        start_idx = valid_len - timesteps_to_mask
        temp_mask[start_idx:] = True  # Mask the first `timesteps_to_mask` elements            
        random_mask[i, :valid_len] = temp_mask  # Apply the contiguous mask to the valid region
    audio_mask = ~random_mask  # Keep valid positions, excluding masked ones (B,S), the random mask has true for the masked positions and gets inverted here to false
    return txt_padded, tgt_padded, audio_mask, attn_mask

#Prepare data and dataloader
data_keys = list(data_dict.keys())
data_keys = [data_keys[i] for i in torch.randperm(len(data_keys))]
dataset = TTSDataSet(data_keys, training_mels, training_text, max_timesteps)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch), pin_memory=True)

# Function to load model and optimizer checkpoint
def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path,weights_only=True)
    # Check if 'model_state_dict' exists, otherwise assume checkpoint is the state dict itself
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint  # Assume entire checkpoint is the model's state dict 
    # Handle `_orig_mod.` prefix if present in checkpoint keys
    if '_orig_mod.' in next(iter(model_state_dict.keys()), ''):
        model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict.items()}   
    # Load the model state dictionary
    model.load_state_dict(model_state_dict, strict=True)
    # Load optimizer state if present in the checkpoint
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', None)
    else:
        start_epoch, loss = 0, None  # Default values if optimizer state is not available
    
    return model, optimizer, start_epoch, loss

output_dir = "/home/kunit17/Her/Data/TrainingOutput"
# Initialize model, tokenizer, and optimizer
model = Transformer(char_size, d_model, n_heads, n_layers, n_mels, dropout_rate, device) 
model = model.to(device)
#model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
print(sum(p.numel() for p in model.parameters()), 'M parameters')
torch.set_float32_matmul_precision('high')
# Training loop

# Load checkpoint if exists
checkpoint_path = f"/home/kunit17/Her/model_checkpoint/checkpoint_epoch_2200.pth"  # Replace with actual checkpoint path
if os.path.exists(checkpoint_path):
    model, optimizer, start_epoch, previous_loss = load_checkpoint(model, optimizer, checkpoint_path)
    if previous_loss is not None:
        print(f"Resuming training from epoch {start_epoch + 1}, previous loss: {previous_loss}")
    else:
        print(f"Resuming training from epoch {start_epoch + 1}, no previous loss recorded.")
else:
    start_epoch = 0
    print("Starting training from scratch")

mini_batch_size = 302464 # close to E2TTS 302,700
assert mini_batch_size % (batch_size*max_timesteps) == 0
num_micro_epochs = mini_batch_size // (batch_size * max_timesteps)
output_dir = f'./model_checkpoint'
log_dir = "./tensorboard_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

for epoch in range(epochs):
    t0 = time.time()
    last_step = (epoch==epochs-1)    
    epoch_loss = 0.0
    saved = False  # Ensure only one mel_output is saved per epoch
    model.train()
    optimizer.zero_grad()
    loss_accum = 0
    mini_batch_tracker = 0 
    for txt, audio_targets, audio_mask, attn_mask in dataloader:

        txt = txt.to(device, non_blocking=True)
        x1 = audio_targets.to(device, non_blocking=True)  #target distribution
        audio_mask = audio_mask.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        t = torch.rand(batch_size, device=device).view(batch_size,1,1) #(B,1,1)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            vt, flow = model(x1, t, txt, audio_mask=audio_mask, attn_mask=attn_mask)
            loss = nn.functional.mse_loss(vt,flow, reduction='none') #returns element-wise squared diff for each pair
        infill_target = ~audio_mask #infill target is what the model is conditioned on 
        loss = loss * infill_target.unsqueeze(-1)
        loss = loss.sum() / (infill_target.sum() * vt.shape[-1]) # need to factor in total elements - check that loss.sum matches attn_mask elements Number of
        loss = loss / num_micro_epochs
        loss_accum += loss.detach() 
        loss.backward()  
                  
        mini_batch_tracker += 1
        if mini_batch_tracker == num_micro_epochs:
            break

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
    lr = learning_rate #get_lr(step) #determine and set learning rate for this iteration 0.0001
    # Log metrics to TensorBoard
    writer.add_scalar("Loss/Train", loss_accum.item(), epoch)
    writer.add_scalar("LearningRate", lr, epoch)
    writer.add_scalar("GradientNorm", norm, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1-t0
    fps = max_timesteps*num_micro_epochs   
    print(f"step {epoch:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | frames/sec: {fps:.2f}") 

    if epoch % 100 == 0 or epoch == epochs - 1:  # Save every 100 epochs and at the last epoch
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_accum.item()
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

writer.close()

#Inference Loops
device = 'cuda' if torch.cuda.is_available() else 'cpu'
time_steps = torch.linspace(0,1, 17).to(device)
txt = "That did the trick. I miss you already, baby"
model.eval()
tokenizer = Tokenizer(chars)
txt = txt.lower()  # Extract the text
encoded_tensor = tokenizer.encode(txt).unsqueeze(0)  # Tokenize and encode directly
txt = torch.nn.functional.pad(encoded_tensor, (0, max_timesteps - encoded_tensor.size(1)), value=4)  #this adds special filler tokens "FILL" to align txt text (S,) dimension with tgt audio dimension (S,)
xt = torch.randn(max_timesteps, n_mels)
txt = txt.to(device)
xt = xt.unsqueeze(0).to(device)
for i in range(16):
    xt = model.step(txt=txt, x_t=xt, t_start = time_steps[i], t_end = time_steps[i+1])

xt = xt.permute(0,2,1)
print(xt.shape)

save_path = "/home/kunit17/Her/Data/TrainingOutput/xt_tensor.pt"
torch.save(xt, save_path)
print(f"Tensor saved at: {save_path}")

