import json
from transformers import BartTokenizer

def create_tokenizer():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    with open("./added_tokens.json", encoding="utf-8") as json_f:
        extra_vocabs = json.load(json_f)
    extra_vocabs = list(extra_vocabs.keys())
    
    tokenizer.add_tokens(extra_vocabs)
    return tokenizer
