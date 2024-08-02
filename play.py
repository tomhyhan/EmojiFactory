from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from torch.utils.data import DataLoader

from data import load_emoji_data, CreateData
from position_encoder import position_encoding_sinusoid
from etokenizer import create_tokenizer

analyzer = SentimentIntensityAnalyzer()
tokenizer = create_tokenizer()

Xy_train, Xy_validation = load_emoji_data(train_size=0.1)

# model params
emb_dim = 32

# trainer params
batch_size = 64

train_data = CreateData(Xy_train, emb_dim, tokenizer, position_encoding_sinusoid, analyzer)
validation_data = CreateData(Xy_validation, emb_dim, tokenizer, position_encoding_sinusoid, analyzer)

train_set = DataLoader(train_data, batch_size, drop_last=True)
validation_set = DataLoader(validation_data, batch_size, drop_last=True)

t = next(iter(train_set))

print(t[4].shape)

