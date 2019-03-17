from sys import exit
from time import ctime
from helpers.data import read_train_dataset, read_categories, text_field
from helpers.batch import BatchWrapper
from helpers.train import train
import torchtext.data as data


if __name__ != '__main__':
    print("This module must be run as the main module.")
    exit(1)
print("Begin the execution at", ctime(), "...\n")

print("Begin to read the categories at", ctime(), "...")
all_categories = read_categories()
print(all_categories)
print("Finished loading all categories at", ctime(), "...\n")

print("Begin to read the training dataset at", ctime(), "...")
train_dataset, val_dataset = read_train_dataset()
print(train_dataset.fields)
print("Finished loading the training dataset at", ctime(), "...\n")

print("Begin to build the vocabulary map at", ctime(), "...")
text_field.build_vocab(train_dataset)
print(len(text_field.vocab))
print("Finished building the vocabulary map at", ctime(), "...\n")

print("Begin to build an iterator at", ctime(), "...")
train_iter, val_iter = data.BucketIterator.splits(
    (train_dataset, val_dataset),
    batch_sizes=(25, 25),
    device=-1,
    sort_key=lambda x: len(x.title),
    sort_within_batch=False,
    repeat=False,
)
train_loader = BatchWrapper(train_iter, "title", ["Category"])
val_loader = BatchWrapper(val_iter, "title", ["Category"])
print("Finished building an iterator at", ctime(), "...")

print("Begin to train the model at", ctime(), "...")
train(train_loader=train_loader, val_loader=val_loader)
print("Finished training the model at", ctime(), "...")
