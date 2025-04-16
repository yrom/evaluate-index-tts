# -*- coding: utf-8 -*-

"""
Evaluate the model on the test set.
"""

import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import List, Literal
import numpy as np
import torch
import torchaudio

from tqdm import tqdm

# Suppress warnings from tensorflow and other libraries
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from indextts.infer import IndexTTS
from indextts.utils.feature_extractors import MelSpectrogramFeatures


@dataclass
class AudioPrompt:
    name: str
    url: str
    lang: Literal["zh", "en"]

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            url=data["url"],
            lang=data["lang"],
        )


@dataclass
class TestCase:
    audio_prompt: AudioPrompt
    text: str


@dataclass
class Text:
    text: str
    lang: Literal["zh", "en"]


@dataclass
class DataSets:
    audio_prompts: List[AudioPrompt]
    short_texts: List[Text]
    long_texts: List[Text]
    extra_texts: List[Text]

    @classmethod
    def from_dict(cls, data):
        audio_prompts = [AudioPrompt.from_dict(d) for d in data.get("audio_prompts", [])]
        zh_texts = data.get("zh")
        en_texts = data.get("en")
        short_texts = [Text(text=t, lang="zh") for t in zh_texts.get("short_texts", [])] + [
            Text(text=t, lang="en") for t in en_texts.get("short_texts", [])
        ]
        long_texts = [Text(text=t, lang="zh") for t in zh_texts.get("long_texts", [])] + [
            Text(text=t, lang="en") for t in en_texts.get("long_texts", [])
        ]
        extra_texts = [Text(text=t, lang="zh") for t in zh_texts.get("extra_texts", [])] + [
            Text(text=t, lang="en") for t in en_texts.get("extra_texts", [])
        ]
        return cls(
            audio_prompts=audio_prompts,
            short_texts=short_texts,
            long_texts=long_texts,
            extra_texts=extra_texts,
        )

    def as_testset(
        self,
        lang: Literal["zh", "en", "all"],
        text: Literal["short", "long", "extra", "all"],
    ) -> tuple[List[AudioPrompt], List[str]]:
        """
        Generate test cases based on the specified language and text type.

        Args
            lang (str): Language of the audio prompt. Can be 'zh', 'en', or 'all'.
            text (str): Type of text. Can be 'short', 'long', 'extra', or 'all'.

        Returns
            List[TestCase]: List of test cases.
        """
        if lang == "all":
            audio_prompts = self.audio_prompts
        else:
            audio_prompts = [ap for ap in self.audio_prompts if ap.lang == lang]

        if text == "short":
            texts = [t.text for t in self.short_texts if t.lang == lang or t.lang == "all"]
        elif text == "long":
            texts = [t.text for t in self.long_texts if t.lang == lang or t.lang == "all"]
        elif text == "extra":
            texts = [t.text for t in self.extra_texts if t.lang == lang or t.lang == "all"]
        else:
            texts = [
                t.text
                for t in self.short_texts + self.long_texts + self.extra_texts
                if t.lang == lang or t.lang == "all"
            ]

        return (
            audio_prompts,
            texts,
        )


def load_dataset(test_set_json_path: str) -> DataSets:
    """
    Load the test set from a JSON file.

    Args
        test_set_json_path (str): Path to the JSON file containing test samples.

    Returns
       DataSets: A DataSets object containing the test samples.
    """
    with open(test_set_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return DataSets.from_dict(data)


def prepare_prompts(test_set: DataSets):
    os.makedirs("prompts", exist_ok=True)

    for test_case in tqdm(test_set.audio_prompts):
        audio_mel_path = os.path.join("prompts", test_case.lang, test_case.name + ".npy")
        if os.path.exists(audio_mel_path):
            print(f"Audio mel '{test_case.name}.npy' already exists, skipping.")
            continue
        audio = load_audio_mel(test_case, device="cpu")
        del audio


def load_audio_mel(audio: AudioPrompt, device):
    audio_mel_path = os.path.join("prompts", audio.lang, audio.name + ".npy")
    if os.path.exists(audio_mel_path):
        try:
            # print(f"Load from {audio_mel_path}")
            cond_mel = np.load(audio_mel_path)
            cond_mel = torch.from_numpy(cond_mel)
            if device and cond_mel.device != device:
                cond_mel = cond_mel.to(device)
            return cond_mel
        except Exception as e:
            print(f"Failed to load prebuilt {audio_mel_path}")
            print("Removing the corrupted file.")
            os.remove(audio_mel_path)
    audio_path = os.path.join("prompts", audio.lang, audio.name + ".wav")
    if not os.path.exists(audio_path):
        try:
            from urllib.request import urlretrieve
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            urlretrieve(audio.url, audio_path)
            # print(f"Download '{audio.name}' from {audio.url}")
        except Exception as e:
            import requests
            response = requests.get(audio.url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"}, stream=True)
            if response.status_code == 200:
                with open(audio_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                raise ValueError(f"Failed to download {audio.url}: {response.status_code}")

    # print(f"Load from {audio_path}")
    audio, sr = torchaudio.load(audio_path)
    audio = torch.mean(audio, dim=0, keepdim=True)
    if device:
        audio = audio.to(device)
    if audio.shape[0] > 1:
        audio = audio[0].unsqueeze(0)
    if sr != 24000:
        resample = torchaudio.transforms.Resample(sr, 24000)
        if device:
            resample = resample.to(device)
        audio = resample(audio)
        print(f">> Audio resample from {sr} to 24000")
    mel_spec = MelSpectrogramFeatures()
    if device:
        mel_spec = mel_spec.to(device)
    cond_mel: torch.Tensor = mel_spec(audio)
    if device and cond_mel.device != device:
        cond_mel = cond_mel.to(device)
    np.save(audio_mel_path, cond_mel.cpu().numpy().astype(np.float32))
    print(f"cond_mel shape: {cond_mel.shape}", "dtype:", cond_mel.dtype)
    del audio
    return cond_mel


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
    results = []
    with torch.inference_mode():
        prompts, texts = test_sets
        total_iterations = len(prompts) * len(texts)
        with tqdm(total=total_iterations, desc="Inference Progress") as pbar:

            for prompt in prompts:
                audio_prompt = load_audio_mel(prompt, model.device)

                for text in texts:
                    model.stats = {}
                    normalized_text = model.preprocess_text(text)
                    start_time = time.perf_counter()
                    audio, sr = model.infer_e2e(audio_prompt, normalized_text, verbose=verbose)
                    end_time = time.perf_counter()
                    infer_duration = end_time - start_time
                    audio_length = audio.shape[1] / sr

                    # Generate audio
                    if output_dir:
                        output_path = os.path.join(output_dir, f"spk_{int(end_time)}.wav")
                        torchaudio.save(output_path, audio.cpu(), sr)
                    else:
                        output_path = None
                    # Save results
                    results.append(
                        {
                            "audio_prompt": prompt.name,
                            "text": text,
                            "output_path": output_path,
                            "audio_length": audio_length,
                            "rtf": infer_duration / audio_length,
                            **model.get_stats(),
                        }
                    )
                    pbar.update(1)

    # Save results to csv
    report = os.path.join(
        output_dir,
        "eval_{}_{}results_{}.csv".format(
            model.device, "fp16_" if model.is_fp16 else "", time.strftime("%Y%m%d-%H%M%S")
        ),
    )
    csv_header = results[0].keys()
    from csv import writer

    with open(report, "w", encoding="utf-8") as f:
        cvs_writer = writer(f)
        cvs_writer.writerow(csv_header)
        for result in results:
            cvs_writer.writerow([result[key][:10] if isinstance(result[key], str) else result[key] for key in csv_header])
    print(f"Evaluation results saved to {report}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the model on the test set.")

    subparser = parser.add_subparsers(dest="command", required=True)
    prepare = subparser.add_parser("prepare", help="Prepare the test set.")
    prepare.add_argument("test_set", type=str, help="Path to the test set JSON file.")
    eval = subparser.add_parser("eval", help="Evaluate the model on the test set.")

    eval.add_argument("--model_dir", type=str, required=True, help="Path to the indextts model checkpoints directory.")
    eval.add_argument("--cfg_path", type=str, required=True, help="Path to the indextts model config file.")
    eval.add_argument("--test_set", type=str, required=True, help="Path to the test set JSON file.")
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
    eval.add_argument(
        "--limit", type=int, default=None, help="Limit the number of test samples to evaluate."
    )
    args = parser.parse_args()

    command = args.command
    # Load test set
    if not os.path.exists(args.test_set):
        raise ValueError(f"Test set file {args.test_set} does not exist.")
    test_set = load_dataset(args.test_set)
    if command == "prepare":
        prepare_prompts(test_set)
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
            args.device = "cuda"
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

    # Evaluate model
    evaluate_model(model, test_sets, output_dir=args.output_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()
