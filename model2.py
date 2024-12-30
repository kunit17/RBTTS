import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtune.modules import RotaryPositionalEmbeddings
import jaxtyping
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, einsum, pack, unpack
import einx
from torch import Tensor
from x_transformers.x_transformers import RotaryEmbedding
from x_transformers import AdaptiveRMSNorm

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]

Float = TorchTyping(jaxtyping.Float)
Int   = TorchTyping(jaxtyping.Int)
Bool  = TorchTyping(jaxtyping.Bool)


class DepthwiseConv(nn.Module):
    def __init__(
        self,
        dim, # number of input and output channels - since its depthwise, input=output
        *,
        kernel_size, # size of convolution kernel
        groups = None # number of groups for convolution, defaults to dim meaning a full depthwise convolution where each input is convolved indep with its own filters
    ):
        super().__init__()
        assert not divisible_by(kernel_size, 2)
        groups = default(groups, dim) # full depthwise conv by default

        self.dw_conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups = groups, padding = kernel_size // 2), #padding here kernl//2 means output tens has same length as input/same padding
            nn.SiLU() #silu activation - sigmoid linear unit
        )

    def forward(
        self,
        x,
        mask = None
    ):
        if exists(mask):
            x = einx.where('b n, b n d, -> b n d', mask, x, 0.)
        x = rearrange(x, 'b n c -> b c n') #rearranged to (batch_size, channels, sequence_length), which is the expected format for 1D convolution in PyTorch.
        x = self.dw_conv1d(x)
        out = rearrange(x, 'b c n -> b n c') #rearranged back
        if exists(mask):
            out = einx.where('b n, b n d, -> b n d', mask, out, 0.)
        return out

class TextAudioCrossCondition(nn.Module):
    def __init__(
        self,
        dim,
        dim_text,
    ):
        super().__init__()
        self.text_to_audio = nn.Linear(dim_text + dim, dim, bias = False)
        nn.init.zeros_(self.text_to_audio.weight)
        self.cond_audio_to_text = cond_audio_to_text
        self.audio_to_text = nn.Linear(dim + dim_text, dim_text, bias = False)
        nn.init.zeros_(self.audio_to_text.weight)

    def forward(
        self,
        audio: Float['b n d'],
        text: Float['b n dt']
    ):
        audio_text, _ = pack((audio, text), 'b n *')
        text_cond = self.text_to_audio(audio_text)
        audio_cond = self.audio_to_text(audio_text)
        return audio + text_cond, text + audio_cond

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
        context = x if context is None else context
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

class CustomConvolution: #need to define and understand
    def __init__(self):
        self.heads = 1
    def forward():
        return

    
class Transformer(nn.Module): 
    
    def __init__ (self, block_size, char_size, d_model, n_heads, dropout_rate, n_layers, n_mels):
        super().__init__()
        self.cproj = nn.Linear(c, d_model)
        self.abs_pos_embed = nn.Embedding(max_seq_len, d_model)
        self.time_mlp = nn.Sequential(CustomConvolution(d_model), nn.Linear(d_model+1, d_model))
        self.dim_head = d_model // n_heads

        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            nn.ModuleList([
                DepthwiseConv(dim, kernel_size=kernel_size),   
                DepthwiseConv(dim, kernel_size=kernel_size),
                Decoder(d_model, n_heads, dropout_rate),
                TextAudioCrossCondition(),
                Decoder()
            ]) for _ in range(n_layers)
        ])
            
    def forward(
        self,
        x: Float['b s c'],
        times: Float['b'],
        text: Float['b s c']
    ): 
        b, s, _ = x.size()
        x = self.cproj(x) # project mel spec into transformer's embedding space
        seq = torch.arange(s, device = device) # create a tensor that will be passed through abs pos embedding layer
        x = x + self.abs_pos_embed(seq) # add abs pos embed to input 
        norm_kwargs = {}
        times = self.time_mlp(times) # pass times through an MLP that takes it from b, -> b,d_model -> b,d_model 
        #need to understand this process much better
        norm_kwargs.update(condition = times) #update kwargs for rand mean square normalization

        for layer_modules in self.layers:
            text_conv, speech_conv, text_attn, cross_condition, audio_attn = layer_modules
            text = text_conv(text, mask = mask) + text
            text_attn_output = text_attn(text)
            text = text_attn_output + text
            x, text_embed = cross_condition(x, text_embed)
            x = x + speech_conv(x, mask=mask)
            audio_attn_output = audio_attn(x)
            x = x + audio_attn_output

        return mel_output, loss






    

        




