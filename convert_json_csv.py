# convert_to_csv.py
import json
import csv

input_file = 'Electronics_5.json'
output_file = 'reviews.csv'

with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout:
    writer = csv.writer(fout)
    writer.writerow(['asin', 'overall'])

    for line in fin:
        try:
            obj = json.loads(line)
            if 'asin' in obj and 'overall' in obj:
                writer.writerow([obj['asin'], obj['overall']])
        except json.JSONDecodeError:
            continue
