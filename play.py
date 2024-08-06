import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from torch.utils.data import DataLoader

from data import load_emoji_data, CreateData
from position_encoder import position_encoding_sinusoid
from etokenizer import create_tokenizer
from train import train_model
from etransformers import EmojiTransformers


analyzer = SentimentIntensityAnalyzer()
tokenizer = create_tokenizer()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

Xy_train, Xy_validation = load_emoji_data(train_size=0.1)

# model params
num_dec_layers = 6
num_enc_layers = 6
num_heads = 4
emb_dim = 32
feedforward_dim = 128
dropout = 0.1
vocab_len = len(tokenizer)

# trainer params
batch_size = 64
num_epochs = 2
loss_func = torch.nn.functional.cross_entropy
lr = 1e-5
warmup_iterations = 100
warmup_lr = 1e-6

train_data = CreateData(Xy_train, emb_dim, tokenizer, position_encoding_sinusoid, analyzer)
validation_data = CreateData(Xy_validation, emb_dim, tokenizer, position_encoding_sinusoid, analyzer)

train_set = DataLoader(train_data, batch_size, drop_last=True)
validation_set = DataLoader(validation_data, batch_size, drop_last=True)

model = EmojiTransformers(num_dec_layers, num_enc_layers, num_heads, emb_dim, feedforward_dim, dropout, vocab_len,  device)

train_model(model, train_set, validation_set, num_epochs, batch_size, loss_func, lr, warmup_iterations, warmup_lr, device)
