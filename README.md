## Evaluate the Index-TTS model

This repository provides a script to evaluate the [Index-TTS](https://github.com/index-tts/index-tts) model

### Prerequisites

```bash
conda create -n index-tts python=3.10
conda activate index-tts
```

### Install torch if not already installed

#### On Windows,

1. Install [CUDA Tools(recommend 12.4)](https://developer.nvidia.com/cuda-downloads) if not already installed. 
2. Install PyTorch with the following command:

```bash
conda install pytorch==2.5.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

3. Install other dependencies:

```bash
conda install -c conda-forge pynini==2.1.5
pip install WeTextProcessing==1.0.3
```

4. If you want to evaluate `Custom CUDA kernal for BigVGN`, install `Visual Studio 2022`.


#### On macOS,

Install PyTorch with the following command:

```bash
pip install torch torchaudio
```


### Run the evaluation

1. Clone the repository:

```bash
git clone https://github.com/yrom/eval-index-tts.git
cd eval-index-tts
pip install -r requirements.txt
```

2. Download Index-TTS model weights:

Download pretrained model from [Huggingface](https://huggingface.co/IndexTeam/Index-TTS) , e.g.:

```bash
huggingface-cli download IndexTeam/Index-TTS \
    bigvgan_discriminator.pth bigvgan_generator.pth bpe.model dvae.pth gpt.pth unigram_12000.vocab \
    --local-dir checkpoints
```

3. prepare the testset.json

```bash
python evaluate.py prepare testset.json
```

4. Run the evaluation:

```bash
python evaluate.py eval --model_dir /path/to/index-tts/checkpoints --cfg_path checkpoints/config.yaml \
    --test_set testset.json --output_dir outputs \
    --lang en --text-type short --device cuda
```

The generated audio files will be saved in the `outputs` directory.

If you want to use `Custom CUDA kernal for BigVGN`, run with `--enable_cuda_kernel`:

```bash
python evaluate.py eval --enable_cuda_kernel ...
```

5. Compare the evaluation results with the baseline:

```bash
python compare.py baseline.csv outputs/xxx.csv
```

## Evaluation Results Sample


Test device: `MacBook Pro M3` `macOS 15.3.2 (24D81)`
Test set: `testset.json`
language: `en`
text type: `short`

device | RTF(lower is better)
---|---
CPU | mean=6.76, std=0.70, min=5.76, max=7.91
MPS | mean=3.87, std=0.57, min=3.16, max=5.39
MPS with fp16 | mean=2.34, std=0.30, min=2.02, max=3.28


## Disclaimer

The audio prompt files featured in `testset.json` were gathered from public sources (refer to the `url` key in json) and are not our property. 
These audio prompts are intended for research purposes and are not commercial use. 
They are provided "as is," without any warranties, guarantees, or representations of any kind. 
By choosing to use these audio prompts, you acknowledge that you do so at your own risk and agree to adhere to all relevant laws and regulations.
We cannot assume responsibility for any damages, losses, or issues that may arise from their use.
Thank you for understanding and respecting these terms!

## License

MIT


