# convert_to_csv.py
import json
import csv

input_file = 'Electronics_5.json'
output_file = 'reviews.csv'
output_file2 = 'reviews2.csv'

with open(input_file, 'r') as fin, open(output_file, 'w', newline='') as fout, open(output_file2, 'w', newline='') as fout2:
    writer = csv.writer(fout)
    writer2 = csv.writer(fout2)
    writer.writerow(['asin', 'overall'])
    writer2.writerow(['reviewerID', 'reviewText'])

    for line in fin:
        try:
            obj = json.loads(line)
            if 'asin' in obj and 'overall' in obj:
                writer.writerow([obj['asin'], obj['overall']])
            if 'reviewerID' in obj and 'reviewText' in obj:
                writer2.writerow([obj['reviewerID'], obj['reviewText']])
        except json.JSONDecodeError:
            continue
