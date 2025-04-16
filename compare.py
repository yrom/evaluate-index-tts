"""
Compare two evaluation results and print the differences.
"""

import csv
import sys

def read_csv(file_path):
    csv_data = []
    with open(file_path, 'r', encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)

        for row in reader:
            result = {}
            for i, v in enumerate(row):
                k = header[i]
                result[k] = v
            csv_data.append(result)
    return csv_data

def calculate_floats(data, key):
    """
    Calculate the mean, std, min, and max of a list of floats.
    """
    values = [float(d[key]) for d in data]
    mean = sum(values) / len(values)
    std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
    min_value = min(values)
    max_value = max(values)
    return mean, std, min_value, max_value

def compare_results(file1, file2):
    data1 = read_csv(file1)
    data2 = read_csv(file2)

    # Compare the two lists of dictionaries

    indicators = ["gpt_gen_time", "gpt_forward_time", "bigvgan_time"]

    for indicator in indicators:
        mean1, std1, min1, max1 = calculate_floats(data1, indicator)
        mean2, std2, min2, max2 = calculate_floats(data2, indicator)
        print(f"{indicator} 1: mean={mean1:.2f}, std={std1:.2f}, min={min1:.2f}, max={max1:.2f}")
        print(f"{indicator} 2: mean={mean2:.2f}, std={std2:.2f}, min={min2:.2f}, max={max2:.2f}")
        print(f"Difference in {indicator}: {abs(mean1 - mean2):.2f}")


    # TODO: Compare the two audio files
    audio1 ={ d["audio_prompt"] + '_' + d['text'] : d["output_path"] for d in data1}
    audio2 ={ d["audio_prompt"] + '_' + d['text'] : d["output_path"] for d in data2}



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare.py <file1.csv> <file2.csv>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    compare_results(file1, file2)