import json
import torchtext.data as data

tokenize = lambda x: x.split()
text_field = data.Field(sequential=True, tokenize=tokenize, lower=True)
label_field = data.Field(sequential=False, use_vocab=False)


def read_categories():
    categories = {}

    with open('../data/categories.json') as file:
        data = json.load(file)
        for category, ID in data["Mobile"].items():
            categories[category] = ID
        for category, ID in data["Fashion"].items():
            categories[category] = ID
        for category, ID in data["Beauty"].items():
            categories[category] = ID

    return categories


def read_dataset(file_path):
    # Declares the fields in the csv file.
    train_fields = [
        ("itemid", None),
        ("title", text_field),
        ("Category", label_field),
        ("image_path", None),
    ]

    # Reads the csv file as a dataset.
    return data.TabularDataset(path=file_path, format="csv", skip_header=True, fields=train_fields)
