from sys import exit
from time import ctime
from src.data import read_dataset, read_categories, text_field
from src.batch import BatchWrapper
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
train_dataset = read_dataset("../data/train.csv")
print(train_dataset.fields)
print("Finished loading the training dataset at", ctime(), "...\n")

print("Begin to build the vocabulary map at", ctime(), "...")
text_field.build_vocab(train_dataset)
print("Finished building the vocabulary map at", ctime(), "...\n")

print("Begin to build an iterator at", ctime(), "...")
train_iter = data.Iterator(
    train_dataset,
    batch_size=64,
    device=-1,
    sort=False,
    sort_within_batch=False,
    repeat=False,
)
train_loader = BatchWrapper(train_iter, "title", ["Category"])
print("Finished building an iterator at", ctime(), "...")

print("Begin to train the model at", ctime(), "...")
print("Finished training the model at", ctime(), "...")
