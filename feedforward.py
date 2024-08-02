import torch
from torch import nn

class FeedForward(nn.Module):
    """
        An implementation of feedforward network 

        input goes through mlp - gelu - mlp
        
    """
    def __init__(self, emb_dim, feedforward_dim):
        """
            inputs:
                emb_dim: embedding dim
                feedforward_dim  
        """
        super().__init__()
        
        self.mlp1 = nn.Linear(emb_dim, feedforward_dim)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(feedforward_dim, emb_dim)

    def forward(self, x):
        """
            inputs:
                x: (N, K, M) N - number of batches, K - number of sequences, M - number of embeddings
            outputs:
                out: (N, K, M)
        """
        out = self.mlp1(x)
        out = self.gelu(out)
        out = self.mlp2(out) 
        return out

