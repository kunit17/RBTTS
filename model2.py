import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from torch import Tensor
from x_transformers.x_transformers import RotaryEmbedding
from x_transformers import AdaptiveRMSNorm
import math



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

class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_heads, d_model, dropout_rate)
        self.ffwd1 = FeedForward(d_model, dropout_rate)
        self.ffwd2 = FeedForward(d_model, dropout_rate)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, decoder_mask=None):
        # Self-attention 
        x = x + self.self_attn(self.ln1(x), context=None, mask=decoder_mask) # (B, timesteps, D_model) skip project
        # Feedforward 
        x = x + self.ffwd2(self.ffwd1(x))
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


    def forward(self, x, context=None, mask=None):    
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
        mask = mask.unsqueeze(1).unsqueeze(2) # (B,1,1,S)
        attention_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False) # Flash attention; no dropout
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
        self.d_model = d_model
        self.xt_proj = nn.Linear(n_mels, d_model)
        self.cond_proj = nn.Linear(2*n_mels+H, d_model) #2C+H
        self.text_embedding_table = nn.Embedding(char_size, H) # H is 256
        self.abs_pos_embed = nn.Embedding(max_timesteps, d_model)
        self.dim_head = d_model // n_heads
        self.register_buffer(
            "freqs",
            torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
            )
        )
        self.n_layers = n_layers

            
    def forward(
        self,
        xt, #: Float['b s c'] noisy speech
        cond, #x_ctx masked speech
        times, #: Float['b'],
        text, #: Float['b s c'],
        audio_mask,
        txt_mask,
        device
    ): 
        text = self.text_embedding_table(text) # B, S -> B, S, H embed dimension
        assert text.size(1) == xt.size(1) == cond.size(1)
        #xt = self.xt_proj(xt) # project mel spec into transformer's embedding space
        #cond = self.cond_proj(cond)
        #xt = xt + cond # add conditioned and interpolated audio inputs for infilling task
        Hc = torch.cat([xt, cond, text], dim=2) #Resulting shape is B,S,(2C+H)
        Hc = self.xt_proj(Hc) # B, S, D_model
        #seq = torch.arange(s, device = device) # create a tensor that will be passed through abs pos embedding layer
        #pos_embed = self.abs_pos_embed(seq)
        #Hc = Hc + self.abs_pos_embed(pos_embed) # add abs pos embed to input 
        args = times[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.d_model % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)



        return xt