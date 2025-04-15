"""
Compare two evaluation results and print the differences.
"""

import csv
import sys

def read_csv(file_path):
    csv_data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)

        for row in reader:
            result = {}
            for i, v in enumerate(row):
                k = header[i]
                result[k] = v
            csv_data.append(result)
    return csv_data


def compare_results(file1, file2):
    data1 = read_csv(file1)
    data2 = read_csv(file2)

    # Compare the two lists of dictionaries
    rtf1 = [float(d['rtf']) for d in data1]
    rtf2 = [float(d["rtf"]) for d in data2]

    rtf1_mean = sum(rtf1) / len(rtf1)
    rtf2_mean = sum(rtf2) / len(rtf2)
    rtf1_std = (sum((x - rtf1_mean) ** 2 for x in rtf1) / len(rtf1)) ** 0.5
    rtf2_std = (sum((x - rtf2_mean) ** 2 for x in rtf2) / len(rtf2)) ** 0.5
    rtf1_min = min(rtf1)
    rtf2_min = min(rtf2)
    rtf1_max = max(rtf1)
    rtf2_max = max(rtf2)

    print(f"RTF1: mean={rtf1_mean:.2f}, std={rtf1_std:.2f}, min={rtf1_min:.2f}, max={rtf1_max:.2f}")
    print(f"RTF2: mean={rtf2_mean:.2f}, std={rtf2_std:.2f}, min={rtf2_min:.2f}, max={rtf2_max:.2f}")

    if rtf1_mean - rtf2_mean > 0.01:
        print("The second results has a lower mean RTF. ")
    elif rtf2_mean - rtf1_mean > 0.01:
        print("The first results has a lower mean RTF.")
    else:
        print("The means are almost same.")

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