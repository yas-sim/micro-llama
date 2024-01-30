import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from collections import OrderedDict

class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape))) # Registering a learnable parameter 'scale' as a parameter of the module

    def forward(self, x):                                                       # Assumes shape is (batch, seq_len, d_model)
        ff_rms = torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5          # Calculating the Frobenius norm, RMS = 1/sqrt(N) * Frobenius norm
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)                            # Normalizing the input tensor 'x' with respect to RMS
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw                    # Scaling the normalized tensor using the learnable parameter 'scale'

class RoPEAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)          # Linear transformation for query
        self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)          # Linear transformation for key
        self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)          # Linear transformation for value
        self.R = self.get_rotary_matrix(config['context_window'], config['d_model'])    # Obtain rotary matrix for positional embeddings

    # Generate rotational matrix for RoPE
    def get_rotary_matrix(self, context_window, embedding_dim):                         
        R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
        for position in range(context_window):
            for i in range(embedding_dim//2):                                           
                theta = 10000. ** (-2. * (i - 1) / embedding_dim)                       # Calculate the rotation angle (theta) based on the position and embedding dimension
                m_theta = position * theta                                              # Calculate the rotated matrix elements using sine and cosine functions
                R[position, 2 * i, 2 * i] = np.cos(m_theta)
                R[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
                R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
                R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
        return R

    def forward(self, x, return_attn_weights=False):                                    # x: input tensor of shape (batch, sequence length, dimension)
        b, m, d = x.shape                                                               # batch size, sequence length, dimension

        # Linear transformations for Q, K, and V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Rotate Q and K using the RoPE matrix
        q_rotated = (torch.bmm(q.transpose(0, 1), self.R[:m])).transpose(0, 1)
        k_rotated = (torch.bmm(k.transpose(0, 1), self.R[:m])).transpose(0, 1)

        # Perform scaled dot-product attention
        activations = F.scaled_dot_product_attention(q_rotated, k_rotated, v, dropout_p=0.1, is_causal=True)

        if return_attn_weights:
            attn_mask = torch.tril(torch.ones((m, m)), diagonal=0)                                  # Create a causal attention mask
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1, 2)) / np.sqrt(d) + attn_mask # Calculate attention weights and add causal mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights

        return activations

class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([RoPEAttentionHead(config) for _ in range(config['n_heads'])])   # Create a list of RoPEMaskedAttentionHead instances as attention heads
        self.linear = nn.Linear(config['n_heads'] * config['d_model'], config['d_model'])           # Linear layer after concatenating heads
        self.dropout = nn.Dropout(.1)                                                               # Dropout layer

    def forward(self, x):                       # x: input tensor of shape (batch, sequence length, dimension)
        heads = [h(x) for h in self.heads]      # Process each attention head and concatenate the results
        x = torch.cat(heads, dim=-1)
        x = self.linear(x)                      # Apply linear transformation to the concatenated output
        x = self.dropout(x)                     # Apply dropout
        return x

class SwiGLU(nn.Module):
    """ Paper Link -> https://arxiv.org/pdf/2002.05202v1.pdf """
    def __init__(self, size):
        super().__init__()
        self.linear_gate = nn.Linear(size, size)        # Linear transformation for the gating mechanism
        self.linear = nn.Linear(size, size)             # Linear transformation for the main branch
        self.beta = torch.randn(1, requires_grad=True)  # Random initialization of the beta parameter

        # Using nn.Parameter for beta to ensure it's recognized as a learnable parameter
        self.beta = nn.Parameter(torch.ones(1))
        self.register_parameter("beta", self.beta)

    def forward(self, x):
        # Swish-Gated Linear Unit computation
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)               # Element-wise multiplication of the gate and main branch
        return out

# add RMSNorm and residual connection
class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rms = RMSNorm((config['context_window'], config['d_model']))   # RMSNorm layer
        self.attention = RoPEMaskedMultiheadAttention(config)               # RoPE Masked Multihead Attention layer
        self.feedforward = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
        )                                                                   # Feedforward layer with SwiGLU activation

    def forward(self, x):                                                   # one block of attention
        x = self.rms(x)             # RMS pre-normalization
        x = x + self.attention(x)   # residual connection
        x = self.rms(x)             # RMS pre-normalization
        x = x + self.feedforward(x) # residual connection
        return x

class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])         # Embedding layer for token representations
        self.llama_blocks = nn.Sequential(                                              # Sequential block of LlamaBlocks based on the specified number of layers
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config['n_layers'])])
        )
        self.ffn = nn.Sequential(                                                       # Feedforward network (FFN) for final output
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
            nn.Linear(config['d_model'], config['vocab_size']),
        )
        print("model params:", sum([m.numel() for m in self.parameters()]))             # Print total number of parameters in the model

    def forward(self, idx, targets=None):
        x = self.embeddings(idx)            # Input token indices are passed through the embedding layer
        x = self.llama_blocks(x)            # Process the input through the LlamaBlocks
        logits = self.ffn(x)                # Pass the processed input through the final FFN for output logits
        if targets is None:                 # If targets are not provided, return only the logits
            return logits
        else:                               # If targets are provided, compute and return the cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
