import csv

train = []
validate = []
trainingRatio = 0.8
with open("../data/train.csv") as f:
    trainingFile = csv.reader(f, delimiter=",")
    beauty = []
    mobile = []
    fashion = []
    for row in trainingFile:
        if row[0] == "itemid":
            continue
        category = int(row[2])
        if category <= 16:
            beauty.append(row)
        elif category >= 31:
            mobile.append(row)
        else:
            fashion.append(row)
    all_data = [beauty, mobile, fashion]
    for data in all_data:
        trainSize = int(trainingRatio * (len(data)))
        train += data[:trainSize]
        validate += data[trainSize:]

with open("../data/validate.csv", "w") as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["itemid","title","Category","image_path"])
    for row in validate:
        writer.writerow(row)

with open("../data/actual_train.csv", "w") as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["itemid","title","Category","image_path"])
    for row in train:
        writer.writerow(row)
