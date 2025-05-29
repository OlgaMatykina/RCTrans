# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
# flash-attention 
import math
import torch
import torch.nn as nn
from torch.nn.init import (
    xavier_uniform_,
    constant_,
    xavier_normal_
)
from torch.nn.functional import linear

from einops import rearrange
from mmcv.runner import auto_fp16
from mmcv.runner.base_module import BaseModule

# from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
# from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis


def _in_projection_packed(q, k, v, w, b = None):
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
    """
    def __init__(self, embed_dim, num_heads, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.fp16_enabled = True

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attention_dropout)

    @auto_fp16(apply_to=('q', 'kv'), out_fp32=True)
    def forward(self, q, kv, 
                causal=False, 
                key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, T, H, D) 
            kv: The tensor containing the key, and value. (B, S, 2, H, D) 
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert q.dtype in [torch.float16, torch.bfloat16] and kv.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        assert q.shape[0] == kv.shape[0] and q.shape[-2] == kv.shape[-2] and q.shape[-1] == kv.shape[-1]

        # print('Q SHAPE', q.shape)
        # print('KV SHAPE', kv.shape)

        batch_size = q.shape[0]
        num_head = q.shape[2]
        seqlen_q, seqlen_k = q.shape[1], kv.shape[1]
        # if key_padding_mask is None:
        #     q, kv = rearrange(q, 'b s ... -> (b s) ...'), rearrange(kv, 'b s ... -> (b s) ...')
        #     max_sq, max_sk = seqlen_q, seqlen_k 
        #     cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
        #                             device=q.device)
        #     cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
        #                             device=kv.device)                    
        #     output = flash_attn_unpadded_kvpacked_func(
        #         q, kv, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
        #         self.dropout_p if self.training else 0.0,
        #         softmax_scale=self.softmax_scale, causal=causal
        #     )
        #     output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        # else:
        #     nheads = kv.shape[-2]
        #     q = rearrange(q, 'b s ... -> (b s) ...')
        #     max_sq = seqlen_q
        #     cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
        #                             device=q.device)
        #     x = rearrange(kv, 'b s two h d -> b s (two h d)')
        #     x_unpad, indices, cu_seqlens_k, max_sk = unpad_input(x, key_padding_mask)
        #     x_unpad = rearrange(x_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads)
        #     output_unpad = flash_attn_unpadded_kvpacked_func(
        #         q, x_unpad, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
        #         self.dropout_p if self.training else 0.0,
        #         softmax_scale=self.softmax_scale, causal=causal
        #     )
        #     output = rearrange(output_unpad, '(b s) ... -> b s ...', b=batch_size)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.permute(1, 0)  # (S, B)

        # q, kv = rearrange(q, 'b s ... -> (b s) ...'), rearrange(kv, 'b s ... -> (b s) ...')
        q = rearrange(q, 'b s h d -> s b (h d)')
        k, v = kv.chunk(2, dim=2)
        k, v = rearrange(k, 'b s one h d -> s (b one) (h d)'), rearrange(v, 'b s one h d -> s (b one) (h d)')
        # print('Q SHAPE', q.shape)
        # print('K SHAPE', k.shape)
        # print('V SHAPE', v.shape)
        # Получаем результат внимания
        attn_output, attn_weights = self.attn(q, k, v, key_padding_mask=key_padding_mask)
        # print('OUTPUT SHAPE', attn_output.shape)
        attn_output = rearrange(attn_output, 's b (h d) -> b s h d', h=num_head)
        return attn_output, None


class FlashMHA(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None, **kwargs) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.bias = bias

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        # self.inner_attn = FlashAttention(attention_dropout=attention_dropout, **factory_kwargs)
        self.inner_attn = FlashAttention(embed_dim, num_heads, attention_dropout=attention_dropout, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        
    def forward(self, q, k, v, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        # q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
        kv = torch.stack([k, v], dim=2)
        
        context, attn_weights = self.inner_attn(q, kv, key_padding_mask=key_padding_mask, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights


# import math
# import torch
# import torch.nn as nn
# from torch.nn.init import xavier_uniform_, constant_
# from torch.nn.functional import linear

# from einops import rearrange
# from mmcv.runner import auto_fp16
# from mmcv.runner.base_module import BaseModule

# def _in_projection_packed(q, k, v, w, b=None):
#     w_q, w_k, w_v = w.chunk(3)
#     if b is None:
#         b_q = b_k = b_v = None
#     else:
#         b_q, b_k, b_v = b.chunk(3)
#     return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


# class FlashAttention(nn.Module):
#     """Implement the multi-head attention with softmax (PyTorch version).
#     Arguments
#     ---------
#         embed_dim: The dimensionality of input and output embeddings.
#         num_heads: The number of attention heads.
#         attention_dropout: The dropout rate to apply to the attention (default: 0.1).
#         causal: If True, causal attention is applied (default: False).
#     """
#     def __init__(self, softmax_scale=None, embed_dim, num_heads, attention_dropout=0.0, causal=False, device=None, dtype=None):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.embed_dim = embed_dim
#         self.causal = causal
        
#         # The built-in PyTorch multi-head attention module
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attention_dropout)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)

#         super().__init__()
#         self.softmax_scale = softmax_scale
#         self.dropout_p = attention_dropout
#         self.fp16_enabled = True

#     def forward(self, q, kv, 
#                 causal=False, 
#                 key_padding_mask=None):
#         """Implements the multihead softmax attention.
#         Arguments
#         ---------
#             q: The tensor containing the query. (B, T, H, D)
#             kv: The tensor containing the key, and value. (B, S, 2, H, D)
#             key_padding_mask: a bool tensor of shape (B, S)
#         """
#         batch_size = q.shape[0]
#         seqlen_q, seqlen_k = q.shape[1], kv.shape[1]
#         nheads = q.shape[-2]
#         head_dim = q.shape[-1]

#         # Split KV into key and value
#         k, v = kv.chunk(2, dim=2)  # (B, S, 1, H, D), (B, S, 1, H, D)

#         # Reshape the inputs to match PyTorch attention requirements
#         q = rearrange(q, 'b s h d -> b h s d')  # (B, H, S, D)
#         k = rearrange(k, 'b s one h d -> (b one) h s d')  # (B, H, S, D)
#         v = rearrange(v, 'b s one h d -> (b one) h s d')  # (B, H, S, D)

#         # Calculate the attention scores
#         attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, S, S)

#         if self.softmax_scale is None:
#             self.softmax_scale = 1.0 / (head_dim ** 0.5)

#         # Scale the attention scores
#         attn_scores = attn_scores * self.softmax_scale

#         # Apply the attention mask if provided
#         if key_padding_mask is not None:
#             # Mask the padding positions by setting their attention to a very large negative value
#             attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

#         # Apply causal mask if needed
#         if causal:
#             causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, device=q.device), diagonal=1)
#             attn_scores = attn_scores.masked_fill(causal_mask == 1, float('-inf'))

#         # Apply softmax to get attention weights
#         attn_weights = F.softmax(attn_scores, dim=-1)

#         # Apply dropout if needed
#         attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)

#         # Compute the attention output
#         output = torch.matmul(attn_weights, v)  # (B, H, S, D)

#         # Reshape the output back to the original shape
#         output = rearrange(output, 'b h s d -> b s h d')  # (B, S, H, D)

#         return output, attn_weights


# class FlashMHA(nn.Module):
#     def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
#                  causal=False, device=None, dtype=None, **kwargs) -> None:
#         assert batch_first
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.causal = causal
#         self.bias = bias

#         self.num_heads = num_heads
#         assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
#         self.head_dim = self.embed_dim // num_heads
#         assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

#         self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
#         if bias:
#             self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
#         else:
#             self.register_parameter('in_proj_bias', None)
        
#         # Replace FlashAttention with StandardAttention
#         self.inner_attn = FlashAttention(embed_dim, num_heads, attention_dropout, causal, **factory_kwargs)
#         self._reset_parameters()

#     def _reset_parameters(self) -> None:
#         xavier_uniform_(self.in_proj_weight)
#         if self.in_proj_bias is not None:
#             constant_(self.in_proj_bias, 0.)
        
#     def forward(self, q, k, v, key_padding_mask=None):
#         """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
#            key_padding_mask: bool tensor of shape (batch, seqlen)
#         """
#         q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
#         q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
#         k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
#         v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
#         kv = torch.stack([k, v], dim=2)
        
#         context, attn_weights = self.inner_attn(q, kv, key_padding_mask=key_padding_mask)
#         return context, attn_weights
