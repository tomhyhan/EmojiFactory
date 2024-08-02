import torch
from torch import nn
from encode import Encoder
from decode import Decoder
from utils import get_subsequent_mask

class EmojiTransformers(nn.Module):
    """
        Implementation of transformers model that translate text to Emoji, which also encapsulate the sentiment data
    """
    
    def __init__(self, num_dec_layers, num_enc_layers, num_heads, emb_dim, feedforward_dim, dropout, vocab_len, device="cpu", dtype=torch.float32):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_len, emb_dim)
        
        self.encoder = Encoder(num_enc_layers, num_heads, emb_dim+1, feedforward_dim, dropout)        
        self.decoder = Decoder(num_dec_layers, num_heads, emb_dim, feedforward_dim, dropout)        
    
        self.device = device
        self.dtype = dtype
        self.vocab_len = vocab_len
    
    def forward(self, que, que_pos, ans, ans_pos, sentiment):
        """
        Implements forward pass of transformers
        
        N - number of batches
        K - sequence length
        M - embedding dimension
        
        Inputs:
            que: (N, K)
            que_pos: (N, K, M)
            ans: (N, K)
            ans_pos: (N, K, M)
            sentiment: (N, 1)
        """
        _, K = ans.shape
        que_emb = self.embedding(que)
        que_emb_pos = que_emb + que_pos
        
        ans_emb = self.embedding(ans)
        ans_emb_pos = ans_emb[:, :-1] + ans_pos[:, :-1]
        
        sentiment_scores_expanded = sentiment.unsqueeze(1).expand(-1, que_emb.shape[1], -1)
        
        enc_inp = torch.cat((que_emb_pos, sentiment_scores_expanded), dim=2)
        
        enc_out = self.encoder(enc_inp)
        
        mask = get_subsequent_mask(K-1)
        dec_out = self.decoder(ans_emb_pos, enc_out, mask)
        
        return dec_out.reshape(-1, self.vocab_len)
        
        