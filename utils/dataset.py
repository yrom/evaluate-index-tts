import json
import os
from dataclasses import dataclass
from typing import List, Literal

import numpy as np
import torch


@dataclass(frozen=True)
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


def download_audio(audio: AudioPrompt, audio_path: str):
    if os.path.exists(audio_path):
        return
    try:
        from urllib.request import urlretrieve

        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        urlretrieve(audio.url, audio_path)
        print(f"Download '{audio.name}' from {audio.url}")
    except Exception:
        import requests

        response = requests.get(
            audio.url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
            },
            stream=True,
        )
        if response.status_code == 200:
            with open(audio_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            raise ValueError(f"Failed to download {audio.url}: {response.status_code}")


def load_audio_mel(audio: AudioPrompt, device = None):
    audio_mel_path = os.path.join("prompts", audio.lang, audio.name + ".npy")
    if os.path.exists(audio_mel_path):
        try:
            # print(f"Load from {audio_mel_path}")
            mel_spec = np.load(audio_mel_path)
            mel_spec = torch.from_numpy(mel_spec)
            if device:
                print(f"mel_spec shape: {mel_spec.shape}", "dtype:", mel_spec.dtype)
                return mel_spec.to(device)
            return mel_spec
        except Exception:
            print(f"Failed to load prebuilt {audio_mel_path}")
            print("Removing the corrupted file.")
            os.remove(audio_mel_path)
    audio_path = os.path.join("prompts", audio.lang, audio.name + ".wav")
    if not os.path.exists(audio_path):
        download_audio(audio, audio_path)
    melspec = extract_audio_melspec(audio_path, device)
    print(f"melspec shape: {melspec.shape}", "dtype:", melspec.dtype)
    if mel_spec.dim() != 3 or mel_spec.shape[0] != 1 or mel_spec.shape[1] != 100:
        raise ValueError(f"Unexpected tensor shape: {mel_spec.shape}. Expected [1, 100, T]")
    np.save(audio_mel_path, mel_spec.cpu().numpy().astype(np.float32))
    return mel_spec


def extract_audio_melspec_and_save(audio_path: str, audio_mel_path: str):
    mel_spec = extract_audio_melspec(audio_path, device="cpu")
    print(f"mel_spec shape: {mel_spec.shape}", "dtype:", mel_spec.dtype)
    np.save(audio_mel_path, mel_spec.numpy().astype(np.float32))
    del mel_spec


def extract_audio_melspec(audio_path: str, device=None):
    import torchaudio
    from indextts.utils.feature_extractors import MelSpectrogramFeatures

    print(f">> Extract MelSpectrogram from {audio_path}...")
    audio, sr = torchaudio.load(audio_path)
    if audio.shape[0] > 1:  # stereo to mono
        print(">> Audio is stereo, convert to mono")
        audio = torch.mean(audio, dim=0, keepdim=True)
    if device:
        audio = audio.to(device)
    if sr != 24000:
        resample = torchaudio.transforms.Resample(sr, 24000)
        if device:
            resample = resample.to(device)
        audio = resample(audio)
        print(f">> Audio resample from {sr} to 24000")
    mel_spec = MelSpectrogramFeatures()
    if device:
        mel_spec = mel_spec.to(device)
    mel_spec: torch.Tensor = mel_spec(audio)
    if device:
        mel_spec = mel_spec.to(device)
    del audio
    return mel_spec
