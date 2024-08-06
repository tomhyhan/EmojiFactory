import torch
from torch import nn
from feedforward import FeedForward
 
class EncoderBlock(nn.Module):
    """
        Implementation of encoder block
        
        The input goes through series of layers
            1. Multi-head Attention block
            2. Dropout
            3. Residual connection and layer Normalization
            5. Feed Forward block
            6. Dropout
            7. Residual connection and layer Normalization
            
        As mentioned in "Attention is All you need" Paper
    """
    
    def __init__(self, num_heads, emb_dim, feedforward_dim, dropout):
        super().__init__()
        """
            inputs:
                num_heads: number head in multihead attention block
                emb_dim: embedding dimention
                feedforward_dim: number of dimension in feedforward
                dropout: dropout percentage
        """
        self.multiheadattn = nn.MultiheadAttention(emb_dim, num_heads , batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        
        self.feedforward = FeedForward(emb_dim, feedforward_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
            Implements forward pass of the encoder block
            
            inputs:
                x: (N, K, M)
                    N - number of batches, K - sequence length, M - embedding dimension
            outputs:
                out: (N, K, M)
        """        
        out1, self.weights_softmax = self.multiheadattn(x, x, x)
        out1 = self.dropout(out1)
        out2 = self.norm1(x + out1)
        
        out3 = self.feedforward(out2)
        out3 = self.dropout(out3)
        out = self.norm2(out3 + out2)
        return out
    
class Encoder(nn.Module):
    """
        Implementation of Encoder block. It contains multiple encorder blocked with respoect to the num of encoding layers.
    """
    
    def __init__(self, num_enc_layers, num_heads, emb_dim, feedforward_dim, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(num_heads, emb_dim, feedforward_dim, dropout) for _ in range(num_enc_layers)] 
        )
    
    def forward(self, x):
        """
            Implements forward pass for Encoder layer
            Inputs:
                x: (N, K, M)
                    N - number of batches, K - sequence length, M - embedding dimension
            outputs:
                out: (N, K, M) the result of 
        """ 
        out = x.clone()   
        for layer in self.layers:
            out = layer(out) 
        return out