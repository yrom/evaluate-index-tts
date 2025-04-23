# -*- coding: utf-8 -*-

"""
Evaluate the model on the test set.
"""

from functools import lru_cache
import os
import sys
import time
from typing import List
import warnings

import numpy as np
import torch
import torchaudio

from utils.tqdm import tqdm
from utils.dataset import (
    AudioPrompt,
    DataSets,
    download_audio,
    extract_audio_melspec_and_save,
    load_audio_mel,
    load_dataset,
)
# Suppress warnings from tensorflow and other libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from indextts.infer import IndexTTS


def prepare_prompts(test_set: DataSets, save_wav=False):
    os.makedirs("prompts", exist_ok=True)

    for test_case in tqdm(test_set.audio_prompts, desc="Prepare audio prompts", file=sys.stdout):
        audio_path = os.path.join("prompts", test_case.lang, test_case.name + ".wav")
        audio_mel_path = os.path.join("prompts", test_case.lang, test_case.name + ".npy")
        if save_wav and not os.path.exists(audio_path):
            download_audio(test_case, audio_path)
        if not os.path.exists(audio_mel_path):
            extract_audio_melspec_and_save(audio_path, audio_mel_path)


def evaluate_model(model: IndexTTS, test_sets: tuple[List[AudioPrompt], List[str]], output_dir=None, verbose=False):
    """
    Evaluate the model on the test set and save the results.

    Args
        model (IndexTTS)
        test_set (list): List of test samples.
        output_dir (str): Directory to save the evaluation results.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda', 'mps').
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    os.makedirs("prompts", exist_ok=True)

    prompts, texts = test_sets
    total_iterations = len(prompts) * len(texts)
    with torch.inference_mode():

        @lru_cache(maxsize=2)
        def _get_audio_mel(prompt: AudioPrompt):
            return load_audio_mel(prompt).to(model.device)

        # flatten the test cases
        test_cases = [(prompt, text) for prompt in prompts for text in texts]
        assert len(test_cases) == total_iterations

        # warmup
        t = "Ready to go"
        audio_prompt = _get_audio_mel(prompts[0])
        ret, _ = model.infer_e2e(audio_prompt, t)
        assert ret.shape[0] == 1 and ret.shape[1] > 0
        del ret
        for prompt, text in tqdm(test_cases, desc="Inference Progress"):
            audio_prompt = _get_audio_mel(prompt)
            model.stats = {}
            normalized_text = model.preprocess_text(text)
            start_time = time.perf_counter()
            audio, sr = model.infer_e2e(audio_prompt, normalized_text, verbose=verbose)
            audio = audio.cpu()
            end_time = time.perf_counter()
            infer_duration = end_time - start_time
            audio_length = audio.shape[1] / sr

            # Generate audio
            if output_dir:
                output_path = os.path.join(output_dir, f"spk_{int(end_time)}.wav")
                torchaudio.save(output_path, audio, sr)
            else:
                output_path = None
            del audio
            # Save results
            yield {
                "audio_prompt": prompt.name,
                "text": text,
                "output_path": output_path,
                "audio_length": audio_length,
                "rtf": infer_duration / audio_length,
                **model.get_stats(),
            }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the model on the test set.")

    subparser = parser.add_subparsers(dest="command", required=True)
    prepare = subparser.add_parser("prepare", help="Prepare the test set.")
    prepare.add_argument("--test_set", type=str, default="testset.json", help="Path to the test set JSON file.")
    prepare.add_argument("--save_wav", action="store_true", default=True, help="Save the original audio files.")
    eval = subparser.add_parser("eval", help="Evaluate the model on the test set.")

    eval.add_argument("--model_dir", type=str, required=True, help="Path to the indextts model checkpoints directory.")
    eval.add_argument("--cfg_path", type=str, required=True, help="Path to the indextts model config file.")
    eval.add_argument("--test_set", type=str, default="testset.json", help="Path to the test set JSON file.")
    eval.add_argument("--output_dir", type=str, default=None, help="Directory to save the evaluation results.")
    eval.add_argument(
        "--device", type=str, default=None, help="Device to run the model on (e.g., 'cpu', 'cuda', 'mps')."
    )
    eval.add_argument("--enable_cuda_kernel", action="store_true", help="Enable custom CUDA kernel for BigVGAN.")
    eval.add_argument("--fp16", action="store_true", help="Use fp16 for inference.")
    eval.add_argument("--no-fp16", action="store_false", dest="fp16", help="Disable fp16 for inference.")
    eval.add_argument("--verbose", action="store_true", help="Enable verbose mode.")

    eval.add_argument(
        "--lang", type=str, default="zh", choices=["zh", "en", "all"], help="Language of the audio prompt."
    )
    eval.add_argument(
        "--text-type",
        type=str,
        default="zh",
        choices=["short", "long", "extra", "all"],
        help="Type of text to evaluate.",
    )
    eval.add_argument("--limit", type=int, default=None, help="Limit the number of test samples to evaluate.")
    args = parser.parse_args()

    command = args.command
    # Load test set
    if not os.path.exists(args.test_set):
        raise ValueError(f"Test set file {args.test_set} does not exist.")
    test_set = load_dataset(args.test_set)
    if command == "prepare":
        prepare_prompts(test_set, args.save_wav)
        return
    if not command or command != "eval":
        raise ValueError(f"Unknown command: {command}")
    if not os.path.exists(args.test_set):
        raise ValueError(f"Test set file {args.test_set} does not exist.")
    test_set = load_dataset(args.test_set)
    test_sets = test_set.as_testset(lang=args.lang, text=args.text_type)
    if args.limit:
        test_sets = test_sets[0][: args.limit], test_sets[1][: args.limit]
    if args.device is None:
        if torch.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"
    # Load model
    model = IndexTTS(
        cfg_path=args.cfg_path,
        model_dir=args.model_dir,
        is_fp16=args.fp16,
        device=args.device,
        use_cuda_kernel=args.enable_cuda_kernel,
    )

    # Save results to csv
    report = os.path.join(
        args.output_dir,
        "eval_{}_{}_{}{}results_{}.csv".format(
            model.device,
            f"testset_{args.lang}_{args.text_type}",
            "bigvgan_cuda_kernel_" if model.use_cuda_kernel else "",
            "fp16_" if model.is_fp16 else "",
            time.strftime("%Y%m%d-%H%M%S"),
        ),
    )
    # Evaluate model
    from csv import writer
    csv_header = None
    with open(report, "w", encoding="utf-8") as f:
        cvs_writer = writer(f)
        for result in evaluate_model(model, test_sets, output_dir=args.output_dir, verbose=args.verbose):
            if csv_header is None:
                csv_header = result.keys()
                cvs_writer.writerow(csv_header)
            # Write results to CSV
            cvs_writer.writerow([result[key] for key in csv_header])

    print(f"Evaluation results saved to {report}")


if __name__ == "__main__":
    main()
