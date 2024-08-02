import torch
from datasets import load_dataset
from torch.utils.data import Dataset 

def load_emoji_data(validation_size=0.1, train_size=None):
    """
        load emoji data from "KomeijiForce/Text2Emoji"
        
        if train_size is not None, only use train data size
        equal to train_size
        
        Split the train data into train_data and validation_data
        
        Inputs:
            validation_size = percentage of validation data
            train_size = percentage of training data
        Outputs:
            train_data
            validation_data
    """
    ds = load_dataset("KomeijiForce/Text2Emoji")
    
    if train_size:
        num_samples = int(len(ds["train"]) * train_size)
        dataset = ds["train"].shuffle()
        ds["train"] = dataset.select(range(num_samples))
        
    data = ds["train"].train_test_split(test_size=validation_size)

    return data["train"], data["test"]

class CreateData(Dataset):
    """
        Convert and Preprocess the data
    """
    
    def __init__(self, data, emb_dim, tokenizer, pos_enc, sentiment_analyzer):
        """
            Inputs:
                data: input and output sequence data
                emb_dim: number of embedding dimention 
                tokenizer: tokeizer to tokenize the input and output sequence
                pos_enc: positional encoding function
        """
        self.data = data
        self.emb_dim = emb_dim
        self.tokenizer = tokenizer
        self.pos_enc = pos_enc
        self.sentiment_analyzer = sentiment_analyzer
        self.max_length = 64

        self.seq_pos = pos_enc(self.max_length, self.emb_dim)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        output = self.data[idx]["emoji"]
        
        inp_seq = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')["input_ids"]

        out_seq = self.tokenizer(output, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')["input_ids"]
        
        sentiment = torch.tensor([self.sentiment_analyzer.polarity_scores(text)["compound"]])
        
        return inp_seq[0], self.seq_pos[0], out_seq[0], self.seq_pos[0], sentiment 
        
# if "__main__" == __name__:
#     analyzer = SentimentIntensityAnalyzer()
#     Xy_train, Xy_validation = load_emoji_data()
#     tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
#     emoji = "😄"
    

#     n_added = tokenizer.add_tokens(extra_vocabs)
#     print("n_added", n_added)

#     emb_dim = 32
#     train_data = CreateData(Xy_train, emb_dim, tokenizer, position_encoding_sinusoid, analyzer)
    
#     print(train_data[0][4].shape)
  