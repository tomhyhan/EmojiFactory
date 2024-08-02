import torch
from torch import nn
from feedforward import FeedForward

class DecoderBlock(nn.Module):
    """
        Implemention of Decoder block 
        
        the input goes through series of layers
            1. Multi-head Attention block
            2. Dropout
            3. Residual connection and layer Normalization
            1. cross Multi-head Attention block
            2. Dropout
            3. Residual connection and layer Normalization
            5. Feed Forward block
            6. Dropout
            7. Residual connection and layer Normalization
            
        As mentioned in "Attention is All you need" Paper
    """
    
    def __init__(self, num_heads, emb_dim, feedforward_dim, dropout):
        """
            inputs:
                num_heads: number head in multihead attention block
                emb_dim: embedding dimention
                feedforward_dim: number of dimension in feedforward
                dropout: dropout percentage
        """
        super().__init__()
        self.multiheadattn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feedforward_dim) 
        
        self.crossheadattn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)       
        self.norm2 = nn.LayerNorm(feedforward_dim)
        
        self.feedforward = FeedForward(emb_dim, feedforward_dim)
        self.norm3 = nn.LayerNorm(feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

        self.proj_sentiment_to_emb = nn.Linear(self.emb+1, self.emb)
        
    def forward(self, dec_inp, enc_inp, mask):
        """
            Implements forward pass of the encoder block
            
            inputs:
                x: (N, K, M)
                    N - number of batches, K - sequence length, M - embedding dimension
            outputs:
                out: (N, K, M)
        """        
        enc_inp1 = self.proj_sentiment_to_emb(enc_inp)
        
        out1, self.weights_softmax = self.multiheadattn(dec_inp, dec_inp, dec_inp, attn_mask=mask)
        out1 = self.dropout(out1)
        out2 = self.norm1(dec_inp + out1)
        
        out3, self.weights_softmax_cross = self.crossheadattn(out2, enc_inp1, enc_inp1)
        out3 = self.dropout(out3)
        out4 = self.norm2(out3 + out2)
        
        out5 = self.feedforward(out4)
        out5 = self.dropout(out5)
        out = self.norm3(out5 + out4)
        
        return out
    
    
class Decoder(nn.Module):
    """
        Implementation of Decoder layer that contains multiple Decoderblocks with respect to num of decoding layers
    
        Then, as a last step, run through anoter FC layer project the output emb dimention to the length of vocab 
    """
    
    def __init__(self, num_dec_layers, num_heads, emb_dim, feedforward_dim, dropout, vocab_len):
        self.layer = nn.ModuleList(
            [DecoderBlock(num_heads, emb_dim, feedforward_dim, dropout) for _ in range(num_dec_layers)]
        )
        
        self.proj_to_vocab = nn.Linear(emb_dim, vocab_len)
    
    def forward(self, dec_inp, enc_inp, mask=None):
        out = dec_inp.clone()
        for layer in self.layers:
            out = layer(out, enc_inp, mask)
        
        out = self.proj_to_vocab(enc_inp)
        return out