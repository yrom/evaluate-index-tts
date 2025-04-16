import csv
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
import torchaudio

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

def plot_waveform(waveform: np.ndarray, sample_rate: int, fig_save_path=None, show=True):
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels * 2, 1, figsize=(6, 3 * num_channels))
    wav_axes = axes[:num_channels]
    wavspec_axes = axes[num_channels : ]
    # spec_axes = axes[num_channels * 2 :]

    for c in range(num_channels):
        wav_axes[c].plot(time_axis, waveform[c], linewidth=1)
        wav_axes[c].set_xlim([0, time_axis[-1]])
        wav_axes[c].grid(True)

        cax = wavspec_axes[c].specgram(waveform[c], Fs=sample_rate)
        # figure.colorbar(cax[-1], ax=wavspec_axes[c])
        # im = spec_axes[c].imshow(librosa.power_to_db(specgram[c]), origin="lower", aspect="auto")
        # figure.colorbar(im, ax=spec_axes[c], format="%+2.0f dB")
        # if num_channels > 1:
        #     wav_axes[c].set_ylabel(f"#{c + 1}")
        #     wavspec_axes[c].set_ylabel(f"#{c + 1}")
        #     spec_axes[c].set_ylabel(f"#{c + 1}")
        # if c == 0:
        #     wav_axes[c].set_title("Original Waveform")
        #     wavspec_axes[c].set_title("Original Spectrogram")
        #     spec_axes[c].set_title("Featured Spectrogram")
    figure.tight_layout()
    if fig_save_path:
        figure.savefig(fig_save_path)
    if show:
        plt.show(block=False)
    plt.close(figure)


def generate_html_report(result_dict, merge_key, baseline_name, files):
    from jinja2 import Template
    # Define the HTML template
    template = Template("""
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
        <a href="https://github.com/yrom/evaluate-index-tts>">Get evaluate scripts</a>
        </p>
        </details>
        </header>
        <h2>Baseline and Results</h2>
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
                {% for k, v in result_dict.items() %}
                    <tr>
                        <td>{{ k[0] }}</td>
                        <td>{{ k[1] }}</td>
                        {% for key in merge_key %}
                            {% if key == "output_path" %}
                                <td><audio controls><source src="{{ v['baseline'].get(key, '') }}" type="audio/wav" /></audio>
                                <img src="{{ v['baseline'].get(key, '').replace('.wav', '.png') }}" alt="Waveform" width="200" />
                                </td>
                                {% for file in files %}
                                    <td><audio controls><source src="{{ v.get(file, {}).get(key, '') }}" type="audio/wav" /></audio>
                                    <img src="{{ v[file].get(key, '').replace('.wav', '.png') }}" alt="Waveform" width="200" />
                                    </td>
                                {% endfor %}
                            {% else %}
                                <td>{{ v['baseline'].get(key, '')[:6] }}</td>
                                {% for file in files %}
                                    <td>{{ v.get(file, {}).get(key, '')[:6] }}</td>
                                {% endfor %}
                            {% endif %}
                        {% endfor %}
                    </tr>
                {% endfor %}
            </tbody>
        </table>

    </body>
    </html>
    """)
    import platform
    import psutil
    import torch

    device_info = [
        f"Platform: {platform.system()} {platform.release()} {platform.architecture()[0]}",
        f"Machine: {platform.machine()}",
        f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical",
        f"Memory: {round(psutil.virtual_memory().total / (1024 ** 3), 2)} GB",
        f"CUDA Available: {'Yes' if torch.cuda.is_available() else 'No'}",
    ]

    if torch.cuda.is_available():
        device_info.extend([
            f"CUDA Version: {torch.cuda.__version__}",
            f"GPU: {torch.cuda.get_device_name(0)}",
            f"GPU Memory: {round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)} GB",
        ])
    elif torch.mps.is_available():
        device_info.extend(
            [
                "MPS Available: Yes",
            ]
        )



    return template.render(result_dict=result_dict, merge_key=merge_key, baseline_name=baseline_name, files=files, device_info=device_info)

def main(baseline_csv: str, files: list[str]):
    baseline = read_csv(baseline_csv)
    baseline_name = os.path.splitext(os.path.basename(baseline_csv))[0]
    merge_key = ["output_path", "gpt_gen_time", "gpt_forward_time", "bigvgan_time", "rtf"]

    def prompt_key(d: dict) -> tuple[str, str]:
        return (d["audio_prompt"], d["text"])

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



    # for k, v in result_dict.items():
        # print(f"{k}: {v}")
    # plot waveforms
    output_waveforms = []
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
                # print(f"File exists: {savepath}")
                continue
            waveform, sample_rate = torchaudio.load(audio_path)
            plot_waveform(waveform, sample_rate, fig_save_path=savepath, show=False)
            output_waveforms.append(savepath)
            del waveform
    # write html report
    html_report = generate_html_report(result_dict, merge_key, baseline_name, filenames)
    
    report_name = '_vs_'.join(filenames) + ".html"
    with open(report_name, "w", encoding="utf-8") as f:
        f.write(html_report)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python gen_eval_report.py <baseline_csv> <file1> <file2> ...")
        sys.exit(1)

    baseline_csv = sys.argv[1]
    files = sys.argv[2:]

    main(baseline_csv, files)