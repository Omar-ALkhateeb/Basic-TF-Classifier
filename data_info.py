import csv


height = []
weight = []
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
            # data.append([gender.index(row[0]), row[1], row[2]])
            line_count += 1
    print(f'Processed {line_count} lines.')

print(max(height))
print(max(weight))
