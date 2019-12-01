import csv
import numpy as np


# preprocessing data and saving it


data = []
lbls = []
labelsValues = [
    "Extremely Weak",
    "Weak",
    "Normal",
    "Overweight",
    "Obesity",
    "Extreme Obesity"
]
gender = [
    "Male",
    "Female"
]
height = []
weight = []
sex = []
with open('dataset.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            # print(
            #     f'\t{row[0]},{row[1]},{row[2]},{row[3]}.')
            height.append(int(row[1]))
            weight.append(int(row[2]))
            sex.append(float(gender.index(row[0])))
            # data.append([gender.index(row[0]), row[1], row[2]])
            lbls.append(int(row[3]))
            line_count += 1
    print(f'Processed {line_count} lines.')

for i in range(len(lbls)):
    user = []
    user.append(sex[i])
    user.append(height[i] / max(height))
    user.append(weight[i] / max(weight))
    data.append(user)
    # lbls.append(labelsValues.index(submission["label"]))


data = np.array(data, dtype=np.float32)
labels = np.array(lbls, dtype=np.int8)

print(lbls)


np.savez_compressed("processedData", data=data, labels=labels)

# colors = np.array(cols, dtype=np.float32)
# labels = np.array(lbls, dtype=np.int8)
# np.savez_compressed("processedData", colors=colors, labels=labels)
