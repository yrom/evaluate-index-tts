"""
For macOS, please install ffmpeg 6.0 before running this script.
You can do this by running the following commands:
```
brew install ffmpeg@6
export TORIO_USE_FFMPEG_VERSION=6
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@6/lib
export PATH=/opt/homebrew/opt/ffmpeg@6/bin:$PATH
```
"""

import sys

import argparse
import os
import time
import numpy as np

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
MANUAL_SEED = 22333333


class BooleanOptionalAction(argparse.Action):
    def __init__(self, option_strings, dest, default=None, required=False, help=None):
        noption_strings = []
        for option_string in option_strings:
            noption_strings.append(option_string)
            if option_string.startswith("--"):
                noption_strings.append("--no-" + option_string[2:])
        super(BooleanOptionalAction, self).__init__(
            option_strings=noption_strings,
            dest=dest,
            nargs=0,
            const=None,
            default=default,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string.startswith("--no-"):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)


def main():
    parser = argparse.ArgumentParser(description="IndexTTS profile command line")
    parser.add_argument(
        "testfile",
        type=str,
        default="tests/texts.txt",
        help="Text file containing the sentences to be synthesized. Each line is a sentence.",
    )
    parser.add_argument(
        "-v",
        "--voice",
        type=str,
        default="prompts/zh/Male-4.npy",
        required=True,
        help="Path to the reference voice file (.npy)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="checkpoints/config.yaml",
        help="Path to the config file. Default is 'checkpoints/config.yaml'",
    )
    parser.add_argument(
        "--model_dir", type=str, default="checkpoints", help="Path to the model directory. Default is 'checkpoints'"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save the generated audio files. Default is 'outputs'",
    )
    parser.add_argument(
        "--fp16", default=False, action=BooleanOptionalAction, help="Use FP16 for inference if available"
    )

    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help="Device to run the model on (cpu, cuda, mps). Default is auto-select based on availability",
    )
    parser.add_argument(
        "--profile", action=BooleanOptionalAction, default=False, help="Enable profiling. Default is disabled"
    )
    parser.add_argument("--profile_memory", action=BooleanOptionalAction, help="Profile memory usage")
    parser.add_argument("--profile_with_stack", action=BooleanOptionalAction, help="Profile CPU usage")

    parser.add_argument(
        "--seed", type=int, default=MANUAL_SEED, help="Random seed for reproducibility. Default is " + str(MANUAL_SEED)
    )
    parser.add_argument("--offset", type=int, default=0, help="Offset of the sentences in `testfile` to be processed")
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit the number of sentences from `testfile` to process (0 for no limit)"
    )
    args = parser.parse_args()
    if os.path.exists(args.testfile):
        with open(args.testfile, "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]
    else:
        print(f"Test file {args.testfile} does not exist.")
        parser.print_help()
        sys.exit(1)
    if args.offset > 0:
        lines = lines[args.offset :]
    if args.limit > 0:
        lines = lines[: args.limit]
    if not os.path.exists(args.voice):
        print(f"Audio prompt file {args.voice} does not exist.")
        parser.print_help()
        sys.exit(1)
    trace_dir = os.path.join(args.output_dir, "trace")
    if not os.path.isdir(trace_dir):
        os.makedirs(trace_dir, exist_ok=True)

    from indextts.infer import IndexTTS
    from profileit import profileit, ScheduleArgs

    tts = IndexTTS(cfg_path=args.config, model_dir=args.model_dir, is_fp16=args.fp16, device=args.device)
    sentences = []
    for text in tqdm(lines, desc="PreProcessing Text"):
        text = tts.preprocess_text(text)
        sentences.extend(tts.split_sentences(text))

    with profileit(
        tts,
        trace_report_dir=trace_dir,
        seed=args.seed,
        profile_memory=args.profile_memory,
        with_stack=args.profile_with_stack,
        schedule=ScheduleArgs(
            wait=0,
            warmup=1,
            active=len(sentences),
        ),
        eanble_profile=args.profile,
    ) as (profiled_tts, step_generator):
        output_path = os.path.join(args.output_dir, f"gen_{int(time.time())}.mp3")
        # stream writer
        import torch
        from torchaudio.io import StreamWriter

        try:
            s = StreamWriter(output_path)
        except Exception as e:
            print("Failed to create StreamWriter")
            import platform

            if platform.system() == "Darwin":
                print("brew install ffmpeg@6")
                print("export TORIO_USE_FFMPEG_VERSION=6")
                print("export DYLD_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@6/lib")
                print("export PATH=/opt/homebrew/opt/ffmpeg@6/bin:$PATH")
            raise
        voice_mel = np.load(args.voice)
        voice_mel = torch.from_numpy(voice_mel).to(tts.device)
        infer_time = 0.0
        audio_time = 0.0
        for step, sentence in zip(step_generator, tqdm(sentences, desc="Generating Audio")):
            print(f"Step {step}: {sentence}")
            with torch.inference_mode():
                start_time = time.perf_counter()
                generated_wav, sr = profiled_tts.infer_e2e(voice_mel, normalized_text=sentence)
                audio_time += generated_wav.shape[1] / sr
                infer_time += time.perf_counter() - start_time
                # generated_wav shape: [Channels, Frames]
                # print(f"Generated wav shape: {generated_wav.shape}")
                if not s._is_open:
                    s.add_audio_stream(sample_rate=sr, num_channels=generated_wav.shape[0], format="s16")
                    s.open()
                # convert ot [Frames, Channels]
                # torch.transpose(generated_wav, 0, 1)
                generated_wav = generated_wav.transpose(0, 1)
                s.write_audio_chunk(0, generated_wav.cpu())
                del generated_wav
                # output_path = os.path.join(args.output_dir, f"gen_{step}.wav")
                # torchaudio.save(output_path, generated_wav.cpu(), sr)
        if s._is_open:
            s.close()
        print(f"Generated audio saved to {output_path}")
        print("Sentences:", "-" * 10, *sentences, "-" * 10, sep="\n")
        for k, v in profiled_tts.get_stats().items():
            print(f"{k}: {v:.4f}s")
        print("RTF: ", f"{infer_time / audio_time:.4f}")


if __name__ == "__main__":
    main()
