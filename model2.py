import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from torch import Tensor
from x_transformers.x_transformers import RotaryEmbedding
from x_transformers import AdaptiveRMSNorm
import math
from torchdiffeq import odeint



class FeedForward(nn.Module):
    def __init__(self, d_model, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), # expand 4x more neurons to give model more freedom to learn and expressiveness. (demonstrated by GPT)
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model), # projection
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_heads, d_model, dropout_rate)
        self.ffwd1 = FeedForward(d_model, dropout_rate)
        self.ffwd2 = FeedForward(d_model, dropout_rate)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention 
        x = x + self.self_attn(self.ln1(x), context=None) # (B, timesteps, D_model) skip project
        # Feedforward 
        x = x + self.ffwd2(self.ffwd1(x))
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
    def __init__(self, n_heads, d_model, dropout_rate, max_seq_len=4096):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)  # Projects decoder output to query
        self.k_proj = nn.Linear(d_model, d_model)  # Projects encoder output to key
        self.v_proj = nn.Linear(d_model, d_model)  # Projects encoder output to value
        # Rotary positional embeddings
        self.rotary_pos_emb = RotaryPositionalEmbeddings(dim=self.head_size, max_seq_len=max_seq_len)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):    
        B, T, _ = x.size()
        context = x if context is None else context # in case an encoder is added to future for cross attn
        S = context.shape[1]
        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(context)  # (B, S, d_model)
        v = self.v_proj(context)  # (B, S, d_model)
        # Split into multiple heads and reshape
        q = q.view(B, T, self.n_heads, self.head_size)  # (B, T, n_heads, head_size)
        k = k.view(B, S, self.n_heads, self.head_size)  # (B, S, n_heads, head_size)
        # Apply rotary positional embeddings to query and key Args: x (torch.Tensor): input tensor with shape ``[b, s, n_h, h_d]``
        q = self.rotary_pos_emb(q)
        k = self.rotary_pos_emb(k)     
        q = q.transpose(1, 2)  # (B, n_heads, T, head_size)
        k = k.transpose(1, 2)  # (B, n_heads, S, head_size)
        v = v.view(B, S, self.n_heads, self.head_size).transpose(1,2) # (B, n_heads, S, head_size)
        #mask = mask.unsqueeze(1).unsqueeze(2) # (B,1,1,S)
        attention_output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False) # Flash attention; no dropout
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, d_model)
        # Apply final linear projection
        out = self.dropout(self.proj(attention_output))
        return out
    
class Transformer(nn.Module): 
    
    def __init__ (self, char_size, d_model, n_heads, dropout_rate, n_layers, n_mels, device, max_timesteps):
        super().__init__()
        H = 256 #text embedding lookup dimension
        max_period = 10000 #for sinusoidal embeddings
        half = d_model//2
        self.device = device
        self.d_model = d_model
        self.xt_proj = nn.Linear(n_mels, d_model)
        self.cond_proj = nn.Linear(2*n_mels+H, d_model) #2C+H
        self.text_embedding_table = nn.Embedding(char_size, H) # H is 256
        #self.abs_pos_embed = nn.Embedding(max_timesteps, d_model)
        self.register_buffer(
            "freqs",
            torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).view(1,1,-1)
        )
        self.decoder = Decoder(n_layers, d_model, n_heads, dropout_rate)
        self.NFE = 32  #number of function evaluation - 32 used in Voicebox

    def forward(
        self,
        x1, #: Float['b s c'] target speech
        t, #: Float['b,1,1'],
        text, #: Float['b s c'],
        audio_mask,
    ): 
        x0 = torch.randn_like(x1) #Gaussian noise
        #t = times * (1. - velocity_consistency_delta)
        x_t = (1. - t) * x0 + t * x1 #interpolated training sample/noisy speech
        flow = x1 - x0 #objective
        cond = torch.where(audio_mask.unsqueeze(-1), x1, torch.zeros_like(x1)) #masked speech
        text = self.text_embedding_table(text) # B, S -> B, S, H embed dimension
        assert text.size(1) == xt.size(1) == cond.size(1)
        #xt = self.xt_proj(xt) # project mel spec into transformer's embedding space
        #cond = self.cond_proj(cond)
        #xt = xt + cond # add conditioned and interpolated audio inputs for infilling task
        Hc = torch.cat([x_t, cond, text], dim=2) #Resulting shape is B,S,(2C+H)
        Hc = self.xt_proj(Hc) # B, S, D_model
        #seq = torch.arange(s, device = device) # create a tensor that will be passed through abs pos embedding layer
        #pos_embed = self.abs_pos_embed(seq)
        #Hc = Hc + self.abs_pos_embed(pos_embed) # add abs pos embed to input 
        args = t * self.freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        Hc = torch.cat([Hc, embedding], dim=1) # add sinusoidal time embeddings to Hc
        Hc = self.decoder(Hc)
        return Hc
    
    def __call__(self, sample, cond, t, text):
        cond = torch.randn_like(sample)
        
        return self.forward(xt=sample, cond=cond, t=t, text=text, audio_mask=None)

    
    def inference_step(
            self,
            txt, #
            x_t # 1 , S, C
    )
        t_span = torch.tensor([t_start.item(), t_end.item()]).to(x_t.device)  # Start and end times
        result = odeint(
            func=self,  # The Transformer itself acts as the ODE dynamics
            y0=x_t,  # Initial state
            t=t_span,  # Time span [t_start, t_end]
            method='midpoint',  # Midpoint solver
        )

        # Return the final state at t_end
        return result[-1]
