import glob
import csv
import os

def clean(split):
    reader = csv.reader(open(f"{split}.csv"))
    result = []
    for i, item in enumerate(reader):
        if i == 0:
            header = item
            continue
        filename = "_".join([item[1], item[2].zfill(6), item[3].zfill(6)]) + ".mp4"
        filename = os.path.join(str(split), filename)
        if os.path.isfile(filename):
            result.append(item)
    with open(f"{split}_clean.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for item in result:
            writer.writerow(item)
    print(len(result))

    reader = csv.reader(open(f"{split}_clean.csv"))
    result = []
    for i, item in enumerate(reader):
        if i == 0:
            header = item
            continue
        filename = "_".join([item[1], item[2].zfill(6), item[3].zfill(6)]) + ".mp4"
        filename = os.path.join(str(split), filename)
        if os.path.isfile(filename):
            result.append(item)
    print(len(result))
    print(result[:5])


for i in ["train", "val", "test"]:
    clean(i)

