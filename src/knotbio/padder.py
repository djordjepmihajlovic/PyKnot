# fixes up csv files such that zeros are appended

import csv

knot_choice = "0_1"

def pad_row(row, target_lengths):
    padded_row = row + [0] * (target_lengths - len(row))
    return padded_row

def pad_csv(input_file, output_file, target_lengths):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            padded_row = pad_row(row, target_lengths)
            writer.writerow(padded_row)

def find_longest_row_length(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        max_row_length = max(len(row) for row in reader)
    return max_row_length

max_length = find_longest_row_length(f'../../knot data/dowker_{knot_choice}_padded.csv')

pad_csv(f'../../knot data/dowker_{knot_choice}.csv', f'../../knot data/dowker_{knot_choice}_padded.csv', max_length)