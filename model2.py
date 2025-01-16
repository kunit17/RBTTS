import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from torch import Tensor
import math
import random


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), # expand 4x more neurons to give model more freedom to learn and expressiveness. (demonstrated by GPT)
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * d_model, d_model), # projection
        )

    def forward(self, x):
        return self.net(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_heads, d_model)
        self.ffwd1 = FeedForward(d_model, dropout_rate)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention 
        x = x + self.self_attn(self.ln1(x)) # (B, timesteps, D_model) skip project
        # Feedforward 
        x = x + self.ffwd1(self.ln2(x)) # B
        return x

class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])
        self.n_layers = n_layers
        self.skip_combine = nn.ModuleList([nn.Linear(2*d_model, d_model) for _ in range (n_layers//2)])

    def forward(self, x):
        intermediate_states = []
        for i in range(self.n_layers // 2): #attention loop through first half
            x = self.layers[i](x)
            intermediate_states.append(x) # Store intermediate states        
        for i in range(self.n_layers // 2, self.n_layers): # attention loop for last half with skip connections added
            skip_state = intermediate_states[self.n_layers - i - 1]
            x = torch.cat([x, skip_state], dim = -1)
            x = self.skip_combine[i - self.n_layers // 2](x)
            x = self.layers[i](x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)  # Projects decoder output to query
        self.k_proj = nn.Linear(d_model, d_model)  # Projects encoder output to key
        self.v_proj = nn.Linear(d_model, d_model)  # Projects encoder output to value
        # Rotary positional embeddings
        self.rotary_pos_emb = RotaryPositionalEmbeddings(dim=self.head_size, max_seq_len=512)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None):    
        B, S, _ = x.size()
        q = self.q_proj(x)  # (B, S, d_model)
        k = self.k_proj(x)  # (B, S, d_model)
        v = self.v_proj(x)  # (B, S, d_model)
        # Split into multiple heads and reshape
        q = q.view(B, S, self.n_heads, self.head_size)  # (B, S, n_heads, head_size)
        k = k.view(B, S, self.n_heads, self.head_size)  # (B, S, n_heads, head_size)
        # Apply rotary positional embeddings to query and key Args: x (torch.Tensor): input tensor with shape ``[b, s, n_h, h_d]``
        q = self.rotary_pos_emb(q)
        k = self.rotary_pos_emb(k)     
        q = q.transpose(1, 2)  # (B, n_heads, S, head_size)
        k = k.transpose(1, 2)  # (B, n_heads, S, head_size)
        v = v.view(B, S, self.n_heads, self.head_size).transpose(1,2) # (B, n_heads, S, head_size)
        if attn_mask is not None:   
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2) # B,S - > (B,1,1,S)
        attention_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False) # Flash attention; no dropout
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, S, self.d_model)  # (B, T, d_model)
        # Apply final linear projection
        out = self.proj(attention_output)
        return out
    
class Transformer(nn.Module): 
    
    def __init__ (self, char_size, d_model, n_heads, n_layers, n_mels, dropout_rate, device):
        super().__init__()
        H = 256 #text embedding lookup dimension
        max_period = 10000 #for sinusoidal embeddings
        half = d_model//2
        self.device = device
        self.d_model = d_model
        self.xt_proj = nn.Linear(n_mels, d_model)
        self.cond_proj = nn.Linear(2*n_mels+H, d_model) #2C+H
        self.text_embedding_table = nn.Embedding(char_size, H) # H is 256
        self.register_buffer(
            "freqs",
            torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).view(1,1,-1)
        )
        self.decoder = Decoder(n_layers, d_model, n_heads, dropout_rate)
        self.NFE = 32  #number of function evaluation - 32 used in Voicebox
        self.final_proj = nn.Linear(d_model, n_mels) # ensure that n_mels is what we're going for here

    def forward(
        self,
        x1, #: Float['b s c'] target speech
        t, #: Float['b,1,1'],
        txt, #: Float['b s'],
        audio_mask=None,
        attn_mask=None,
    ): 
        x0 = torch.randn_like(x1) #Gaussian noise
        #t = times * (1. - velocity_consistency_delta)
        x_t = (1. - t) * x0 + t * x1 #interpolated training sample/noisy speech
        flow = x1 - x0 #objective b, s, c
        #Classifer-free guidance - dropping conditioning 20% as per E2TTS
        if self.training and random.random() < 0.2:  # 20% probability during training
            cond = x0  # Use x0 as conditioning
        else:
            cond = torch.where(audio_mask.unsqueeze(-1), x1, torch.zeros_like(x1))  # Masked speech

        cond = torch.where(audio_mask.unsqueeze(-1), x1, torch.zeros_like(x1)) #masked speech -torch.where looks for boolean
        txt = self.text_embedding_table(txt) # B, S -> B, S, H embed dimension
        assert txt.size(1) == x_t.size(1) == cond.size(1)
        Hc = torch.cat([txt, cond, x_t], dim=2) #Resulting shape is B,S,(2C+H)
        Hc = self.xt_proj(Hc) # B, S, D_model
        args = t * self.freqs
        t_embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        Hc = torch.cat([t_embedding, Hc], dim=1) # concat sinusoidal time embeddings to Hc; concat time first -> b, s+1, d_model
        Hc = self.decoder(Hc, attn_mask)
        vt = self.final_proj(Hc) #project transformer output into mel spec b,s,c
        return vt, flow
    
    @torch.no_grad()
    def step(
        self,
        txt, # 1,s
        x_t: Tensor,
        t_start: Tensor, # 1
        t_end: Tensor
    ) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1) # Expand t_start to match the batch size
        delta_t = t_end - t_start # Compute the time difference (Δt)
        f_start = self(x_t=x_t, t=t_start, txt=txt) # Compute the first forward pass (at t_start)
        t_mid = t_start + delta_t / 2 # Compute the midpoint in time
        x_mid = x_t + f_start * delta_t / 2 # Compute the midpoint in state using the first forward pass
        f_mid = self(t=t_mid, x_t=x_mid, txt=txt) # Compute the second forward pass (at the midpoint)
        x_next = x_t + f_mid * delta_t # Update x_t using the midpoint estimate
        #return x_t + (t_end - t_start) * self(t=t_start + (t_end - t_start) / 2, x_t= x_t + self(x_t=x_t, t=t_start) * (t_end - t_start) / 2)
        return x_next



