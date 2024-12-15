import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import Tokenizer 
import utils
from config import chars
from preprocess import data_dict, y
import random
from model import Transformer

torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size, learning_rate, epochs, block_size, char_size, d_model, n_heads, dropout_rate, head_size = utils.get_train_params()
n_layers = 5
n_mels = 128
max_ts = 302


# model = Transformer()
# # m = model.to(device)
# print(sum(p.numel() for p in m.parameters()), 'M parameters')
tokenizer = Tokenizer(chars)
# optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iters in range(5):

    keys = list(data_dict.keys())
    rndm_key = random.choice(keys)
    print(data_dict[rndm_key]['Text'])
    src_idx, encoder_mask = tokenizer.encode(data_dict[rndm_key]['Text'], block_size, device)
    if src_idx.dim() == 1:
        src_idx = src_idx.unsqueeze(0)  # Add batch dimension
    targets = y[rndm_key] 
    if targets.dim() == 2:
        targets = targets.permute(1, 0)  # Ensure (timesteps, n_mels)
    if targets.dim() == 1:
        targets = targets.unsqueeze(0)  # Add batch dimension if needed
    pred = targets.clone()
    # m.train()
    # optimizer.zero_grad(set_to_none=True)
    # logits, loss = m(src_idx, encoder_mask, targets, pred) # get the logit output and calculate loss
    # loss.backward()  # backpropagate
    # optimizer.step()  # update parameters
    # print(f"Iteration {iters + 1}/{epochs}")
    # print(f"Loss: {loss.item()}")
    print(src_idx, encoder_mask, targets)
    break
    
