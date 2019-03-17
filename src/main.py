import json


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


all_categories = read_categories()
print(all_categories)
