import csv
import os

dir_path = r"dataset/Train_submission/Train_submission"

# list to store files
res = []

# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        res.append(path)

sons = []
with open("dataset/Metadata_Train.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            #print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            #print(f'\t{row[0]}')
            sons.append(row[0])
            line_count += 1
    #print(f'Processed {line_count} lines.')

violin = []

for file in res:
    if not (file in sons):
        violin.append(file)


with open('dataset/Metadata_Train2.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for v in violin:
        employee_writer.writerow([v,"Sound_Violin"])

print(len(violin))

