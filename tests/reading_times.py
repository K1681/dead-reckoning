import csv
from datetime import datetime

with open("data.csv" , mode="r") as file:
    csv_file = csv.reader(file)
    initial_time = datetime.strptime(csv_file.__next__()[2].strip(),"%d/%m/%Y %H:%M:%S")
    for line in csv_file:
        next_time = datetime.strptime(line[2].strip(),"%d/%m/%Y %H:%M:%S")
        delta_time = (next_time - initial_time).total_seconds()
        print(delta_time)




        initial_time = next_time
