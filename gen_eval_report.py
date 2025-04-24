import csv
import json
import sys
import os
from typing import Dict, List
import numpy as np
from jinja2 import Template
import time

import psutil

TEMPLATE_FOR_SINGLE_RESULT = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report</title>
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            
        }
        th, td {
            border: 1px solid black;
            padding: 8px;
            text-align: center;
            
        }
        th {
            background-color: #f2f2f2;
        }
        audio {
            width: 150px;
        }
    </style>
</head>
<body>
    <header>
    <h1>Evaluation Report</h1>
    <details><summary>Device Info</summary>
    <p>DeviceInfo: <br/>
    {% for info in device_info %}
        {{ info }}<br/>
    {% endfor %}
    </p>
    <p>
    <a href="https://github.com/yrom/evaluate-index-tts">Get evaluate scripts</a>
    <br/>
    </p>
    </details>
    </header>
    <h2>Evaluation Result</h2>
    <p>Testset: <a href="https://github.com/yrom/evaluate-index-tts/blob/main/testset.json">https://github.com/yrom/evaluate-index-tts/blob/main/testset.json</a> </p>
    <p>Result: <a href="outputs/{{result_name}}.csv" >{{ result_name }}</a></p>
    <table>
        <thead>
            <tr>
                <th>Reference</th>
                <th>Text</th>
                {% for key in merge_key %}
                    <th colspan="{{ files|length + 1 }}">{{ "Generated Audio" if key == "output_path" else key.upper() }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            <tr style="background-color: azure;">
                <td> Average </td>
                <td> - </td>
                {% for key in merge_key %}
                    <td>
                    {% if key == "output_path" %}
                      -
                    {% else %}
                        {{ '%.4f' | format(summary.get(key, 0.0)) }}
                    {% endif %}
                    </td>
                {% endfor %}
            </tr>
            {% for k, v in result_dict.items() %}
                <tr>
                    <td>{{ k[0] }}</td>
                    <td>{{ k[1] }}</td>
                    {% for key in merge_key %}
                    <td>
                        {% set value_of_key = v.get(key, '') %}
                        {% if key == "output_path" %}
                            {% set img_path = value_of_key.replace('.wav', '.png') %}
                            <audio controls preload="none"><source src="{{ value_of_key }}" type="audio/wav" /></audio>
                            <img src="{{ img_path }}" alt="Waveform" width="200" />
                        {% else %}
                            {{ '%.4f' | format( value_of_key | float ) }}
                        {% endif %}
                    {% endfor %}
                    </td>
                </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
"""
TEMPLATE_FOR_MULTI_RESULT = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report</title>
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            
        }
        th, td {
            border: 1px solid black;
            padding: 8px;
            text-align: center;
            
        }
        th {
            background-color: #f2f2f2;
        }
        audio {
            width: 150px;
        }
    </style>
</head>
<body>
    <header>
    <h1>Evaluation Report</h1>
    <details><summary>Device Info</summary>
    <p>DeviceInfo: <br/>
    {% for info in device_info %}
        {{ info }}<br/>
    {% endfor %}
    </p>
    <p>
    <a href="https://github.com/yrom/evaluate-index-tts">Get evaluate scripts</a>
    <br/>
    </p>
    </details>
    </header>
    <h2>Baseline and Results</h2>
    <p>Testset: <a href="https://github.com/yrom/evaluate-index-tts/blob/main/testset.json">https://github.com/yrom/evaluate-index-tts/blob/main/testset.json</a> </p>
    <ul>
        <li>Baseline:  <a href="outputs/{{baseline_name}}.csv" >{{ baseline_name }}</a></li>
        {% for  file in files %}
            <li>Result {{ loop.index }}: <a href="outputs/{{file}}.csv" >{{ file }} </a></li>
        {% endfor %}
    </ul>
    <h2>Evaluation Results</h2>
    <table>
        <thead>
            <tr>
                <th rowspan="2">Reference</th>
                <th rowspan="2">Text</th>
                {% for key in merge_key %}
                    <th colspan="{{ files|length + 1 }}">{{ "Generated Audio" if key == "output_path" else key.upper() }}</th>
                {% endfor %}
            </tr>
            <tr>
                {% for key in merge_key %}
                    <th>Baseline</th>
                    {% for file in files  %}
                        <th>Result {{ loop.index }}</th>
                    {% endfor %}
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            <tr style="background-color: azure;">
                <td> Average </td>
                <td> - </td>
                {% for key in merge_key %}
                    {% if key == "output_path" %}
                        <td>-</td>
                        {% for file in files %}
                            <td>-</td>
                        {% endfor %}
                    {% else %}
                    <td> {{ '%.4f' | format(summary['baseline'].get(key, 0.0)) }} </td>
                    {% for file in files %}
                        <td> {{ '%.4f' | format( summary.get(file, {}).get(key, 0.0)) }} </td>
                    {% endfor %}
                    {% endif %}
                {% endfor %}
            </tr>
            {% for k, v in result_dict.items() %}
                <tr>
                    <td>{{ k[0] }}</td>
                    <td>{{ k[1] }}</td>
                    {% for key in merge_key %}
                        {% if key == "output_path" %}
                            <td><audio controls preload="none"><source src="{{ v['baseline'].get(key, '') }}" type="audio/wav" /></audio>
                            <img src="{{ v['baseline'].get(key, '').replace('.wav', '.png') }}" alt="Waveform" width="200" />
                            </td>
                            {% for file in files %}
                                <td><audio controls preload="none"><source src="{{ v.get(file, {}).get(key, '') }}" type="audio/wav" /></audio>
                                <img src="{{ v[file].get(key, '').replace('.wav', '.png') }}" alt="Waveform" width="200" />
                                </td>
                            {% endfor %}
                        {% else %}
                        
                            <td>{{ '%.4f' | format( v['baseline'].get(key, 0.0) | float ) }}</td>
                            {% for file in files %}
                                <td>{{ '%.4f' | format( v.get(file, {}).get(key, 0.0) | float ) }}</td>
                            {% endfor %}
                        {% endif %}
                    {% endfor %}
                </tr>
            {% endfor %}
        </tbody>
    </table>

</body>
</html>
"""


def read_csv(file_path) -> list[dict]:
    csv_data = []
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)

        for row in reader:
            if not row:
                continue
            result = {}
            for i, v in enumerate(row):
                k = header[i]
                result[k] = v
            csv_data.append(result)
    return csv_data




def list_device_info():
    import platform
    import psutil
    import torch

    device_info = [
        f"Platform: {platform.system()} {platform.release()} {platform.architecture()[0]}",
        f"Machine: {platform.machine()}",
        f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical",
        f"Memory: {round(psutil.virtual_memory().total / (1024**3), 2)} GB",
        f"CUDA Available: {'Yes' if torch.cuda.is_available() else 'No'}",
    ]

    if torch.cuda.is_available():
        device_info.extend(
            [
                f"CUDA Version: {torch.version.cuda}",
                f"CUDA Device Count: {torch.cuda.device_count()}",
            ]
        )
        for i in range(torch.cuda.device_count()):
            device_info.append(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
            device_info.append(
                f"CUDA Device {i} Memory: {round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)} GB"
            )
    elif platform.system() == "Darwin" and hasattr(torch, "mps") and torch.mps.is_available():
        device_info.extend(
            [
                "MPS Available: Yes",
            ]
        )
    return device_info


def generate_html_report_for_single_result(result_name, result_dict, summary, merge_key):
    device_info = list_device_info()
    template = Template(TEMPLATE_FOR_SINGLE_RESULT)
    return template.render(
        result_name=result_name, result_dict=result_dict, summary=summary, merge_key=merge_key, device_info=device_info
    )


def generate_html_report(result_dict, summary, merge_key, baseline_name, files):
    device_info = list_device_info()
    template = Template(TEMPLATE_FOR_MULTI_RESULT)
    return template.render(
        result_dict=result_dict,
        summary=summary,
        merge_key=merge_key,
        baseline_name=baseline_name,
        files=files,
        device_info=device_info,
    )


def plot_waveform_wrapper(params):
    audio_path, savepath = params
    import torchaudio
    from utils.plot import plot_waveform
    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"Plotting waveform for {audio_path} to {savepath}")
    plot_waveform(waveform, sample_rate, fig_save_path=savepath, show=False)
    del waveform


def prompt_key(d: dict) -> tuple[str, str]:
    return (d["audio_prompt"], d["text"])


def report_for_single_result(result_csv: str):
    result = read_csv(result_csv)
    result_name = os.path.splitext(os.path.basename(result_csv))[0]
    merge_key = ["output_path", "gpt_gen_time", "gpt_forward_time", "bigvgan_time", "rtf"]
    result_dict = dict(map(lambda d: (prompt_key(d), {k: d[k] for k in merge_key}), result))
    indicators = ["gpt_gen_time", "gpt_forward_time", "bigvgan_time", "rtf"]
    summary = {}
    for k, vv in result_dict.items():
        values: Dict[str, List[float]] = {}
        for indicator in indicators:
            if indicator not in vv:
                print(f"Missing {indicator} in {result_name} for {k}")
                continue
            vv[indicator] = float(vv[indicator])
            if indicator not in values:
                values[indicator] = []
            values[indicator].append(vv[indicator])
    summary = {indicator: np.mean(value) for indicator, value in values.items()}

    print("Summary of ", result_name)
    print(json.dumps(summary, indent=4))
    # plot waveforms
    plot_params = []
    for vv in result:
        audio_path = vv["output_path"]
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}")
            continue
        savepath = os.path.splitext(audio_path)[0] + ".png"
        if os.path.exists(savepath):
            # print(f"File already exists: {savepath}")
            continue
        plot_params.append((audio_path, savepath))

    # plot waveform in subprocess
    if len(plot_params) > 10:
        import multiprocessing

        num_processes = min(max(4, len(plot_params) // 4), psutil.cpu_count(logical=True))
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(plot_waveform_wrapper, plot_params)
    else:
        for params in plot_params:
            plot_waveform_wrapper(params)

    # write html report
    html_report = generate_html_report_for_single_result(result_name, result_dict, summary, merge_key)

    report_name = f"indextts_evaluate_report_{time.strftime('%Y%m%d_%H%M%S')}.html"
    with open(report_name, "w", encoding="utf-8") as f:
        f.write(html_report)
    print(f"Report saved to {os.path.abspath(report_name)}")


def report_for_multi_results(baseline_csv: str, files: List[str]):
    baseline = read_csv(baseline_csv)
    baseline_name = os.path.splitext(os.path.basename(baseline_csv))[0]
    merge_key = ["output_path", "gpt_gen_time", "gpt_forward_time", "bigvgan_time", "rtf"]

    result_dict = dict(map(lambda d: (prompt_key(d), {"baseline": {k: d[k] for k in merge_key}}), baseline))

    keys = result_dict.keys()
    filenames = [os.path.splitext(os.path.basename(file))[0] for file in files]
    for file in files:
        data = read_csv(file)
        filename = os.path.splitext(os.path.basename(file))[0]
        data_dict = {prompt_key(d): d for d in data}
        # merge the data
        for result_key in keys:
            if result_key not in data_dict:
                print(f"Missing {result_key} in {file}")
                continue
            d = data_dict[result_key]
            result_dict[result_key].update({filename: {k: d[k] for k in merge_key}})

    # summary for the indicators:  "gpt_gen_time", "gpt_forward_time", "bigvgan_time", "rtf"
    indicators = ["gpt_gen_time", "gpt_forward_time", "bigvgan_time", "rtf"]
    summary = {}

    for filename in ["baseline", *filenames]:
        values_of_file: Dict[str, List[float]] = {}
        for k, v in result_dict.items():
            if filename not in v:
                print(f"Missing {filename} in {v} for {k}")
                continue
            vv = v[filename]

            for indicator in indicators:
                if indicator not in vv:
                    print(f"Missing {indicator} in {filename} for {k}")
                    continue
                vv[indicator] = float(vv[indicator])
                if indicator not in values_of_file:
                    values_of_file[indicator] = []
                values_of_file[indicator].append(vv[indicator])

        summary[filename] = {indicator: np.mean(value) for indicator, value in values_of_file.items()}
        print(f"Summary of {filename}")
        print(json.dumps(summary[filename], indent=4))
        print("===" * 10)

    # plot waveforms
    plot_params = []
    for k, v in result_dict.items():
        for filename in ["baseline", *filenames]:
            if filename not in v:
                print(f"Missing {filename} in {v} for {k}")
                continue
            vv = v[filename]
            audio_path = vv["output_path"]
            if not os.path.exists(audio_path):
                print(f"File not found: {audio_path}")
                continue
            savepath = os.path.splitext(audio_path)[0] + ".png"
            if os.path.exists(savepath):
                # print(f"File already exists: {savepath}")
                continue
            plot_params.append((audio_path, savepath))

    # plot waveform in subprocess
    if len(plot_params) > 10:
        import multiprocessing
        num_processes = min(max(4, len(plot_params) // 4), psutil.cpu_count(logical=True))
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(plot_waveform_wrapper, plot_params)
    else:
        from utils.tqdm import tqdm
        for params in tqdm(plot_params):
            plot_waveform_wrapper(params)

    # write html report
    html_report = generate_html_report(result_dict, summary, merge_key, baseline_name, filenames)

    report_name = f"indextts_evaluate_report_{time.strftime('%Y%m%d_%H%M%S')}.html"
    with open(report_name, "w", encoding="utf-8") as f:
        f.write(html_report)
    print(f"Report saved to {os.path.abspath(report_name)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage:",
            "  python gen_eval_report.py <result.csv>",
            "  python gen_eval_report.py <baseline.csv> [file1.csv] [file2.csv] ...",
            sep="\n",
        )
        sys.exit(1)

    baseline_csv = sys.argv[1]
    if not os.path.exists(baseline_csv):
        print(f"File not found: {baseline_csv}")
        sys.exit(1)
    if len(sys.argv) > 2:
        files = sys.argv[2:]
        report_for_multi_results(baseline_csv, files)
    else:
        report_for_single_result(baseline_csv)
