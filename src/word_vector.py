import spacy
import torch
from torchtext import data, datasets
spacy_en = spacy.load('en')

def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]

TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=150)
LABEL = data.Field(sequential=False, use_vocab=False)
train, val, test = data.TabularDataset.splits(
        path='../data/', train='train.csv',
        validation='val.csv', test='test.csv', format='csv',
        fields=[('Text', TEXT), ('Label', LABEL)])
TEXT.build_vocab(train, vectors="glove.6B.100d")
train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test), sort_key=lambda x: len(x.Text),
        batch_sizes=(32, 256, 256), device=-1)
vocab = TEXT.vocab