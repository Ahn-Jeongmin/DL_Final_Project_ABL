echo "Python path: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# 실제 작업 실행
CUDA_VISIBLE_DEVICES=4 
export FLASH_ATTENTION=0
export USE_FLASH_ATTENTION=0
export HUGGINGFACE_USE_FLASH_ATTENTION=0
export HUGGINGFACE_MEGA_USE_FLASH_ATTENTION=0
export XFORMERS_DISABLED=1
python anal.py