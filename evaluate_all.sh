#!/bin/bash
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model_dir>"
    exit 1
fi
# Check if the model directory exists
model_dir=$1
if [[ ! -d $model_dir ]]; then
    echo "Model directory $model_dir does not exist."
    exit 1
fi
model_files=(
    "bigvgan_discriminator.pth"
    "bigvgan_generator.pth"
    "bpe.model"
    "dvae.pth"
    "gpt.pth"
    "unigram_12000.vocab"
)
# Function to download the pretrained model
download_pretrained_model() {
    local downlaod_dir=$1
    echo "Downloading pretrained model..."
    mkdir -p $downlaod_dir
    model_repo="IndexTeam/Index-TTS"
    
    if huggingface-cli env ; then
        export HF_HUB_DOWNLOAD_TIMEOUT=30
        huggingface-cli download $model_repo \
         ${model_files[@]} \   
         --local-dir $downlaod_dir
    else
        echo "huggingface-cli is not installed. use wget instead."
        hf_endpoint="${HF_ENDPOINT:-https://huggingface.co}"
        # Download each file using wget
        for file in "${model_files[@]}"; do
            echo "Downloading $file..."
            wget --content-on-error --no-check-certificate "$hf_endpoint/$model_repo/resolve/main/$file" -P $downlaod_dir
        done
    fi
}


# CUDA_VISIBLE_DEVICES by the free GPU
update_cuda_visible_devices() {
    if command -v nvidia-smi &> /dev/null; then
        # sorted_devices=(0 2 3 1)
        sorted_devices=(`nvidia-smi --query-gpu=index,utilization.gpu,memory.free,memory.total --format=csv,noheader,nounits | sort -k2,2n -k3,3nr | awk '{print $1}' | sed 's/,$//'`)
        savedIFS="$IFS"
        IFS=","
        export CUDA_VISIBLE_DEVICES="${sorted_devices[*]}"
        IFS="$savedIFS"
        echo "Update CUDA_VISIBLE_DEVICES to ${sorted_devices[@]}"
    else
        echo "nvidia-smi command not found. Do not modify CUDA_VISIBLE_DEVICES."
    fi
}

main() {
    # Check if the model directory contains the required files
    for file in "${model_files[@]}"; do
        if [[ ! -f $model_dir/$file ]]; then
            echo "$file not found in $model_dir. Downloading pretrained model..."
            download_pretrained_model $model_dir
            break
        fi
    done

    update_cuda_visible_devices
    # Run the evaluation script
    echo "Running evaluation script..."

    now_date=$(date +"%Y%m%d_%H%M%S")
    # Create a directory to save the outputs
    outputs_dir="outputs_$now_date"
    mkdir -p $outputs_dir
    python evaluate.py prepare
    echo "Evaluating baseline result"

    base_args="--model_dir $model_dir --cfg_path checkpoints/config.yaml \
        --lang zh --text-type all --test_set testset.json --no-fp16 \
        --output_dir $outputs_dir"

    python evaluate.py eval $base_args
    
    update_cuda_visible_devices
    echo "Evaluating fp16 result"
    python evaluate.py eval $base_args --fp16

    # check nvcc and ninja if available
    if command -v nvcc &> /dev/null && nvcc -V; then
        echo "nvcc is available, compiling custom cuda extension"
        if python -c "from indextts.BigVGAN.alias_free_activation.cuda import load; print(load.load())"; then
            echo "Custom cuda extension compiled"
            echo "Evaluating custom cuda extension"
            python evaluate.py eval $base_args --enable_cuda_kernel
        else
            echo "Failed to compile custom cuda extension, skip evaluation."
        fi
    
    else
        echo "nvcc is not available, skip custom cuda extension evaluation."
    fi

    # check deepspeed if installed
    if python -m deepspeed.env_report; then
        echo "Evaluating deepspeed result"
        python evaluate.py eval --model_dir $model_dir --cfg_path checkpoints/config.yaml \
            --lang zh --text-type all --test_set testset.json --fp16 --enable_deepspeed \
            --output_dir $outputs_dir 
    else
        echo "deepspeed not installed, skip deepspeed evaluation."
    fi
    reports=(`ls $outputs_dir/*.csv`)
    python gen_eval_report.py ${reports[@]}
}


# Call the main function
main "$@"