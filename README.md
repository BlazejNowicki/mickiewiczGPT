
# MickiewiczGPT



### Quickstart

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

```sh
python data/pan_tadeusz_char/prepare.py
```

```sh
python train.py config/train_pan_tadeusz_char.py
```

```sh
python sample.py --out_dir=out-pan-tadeusz-char
```

## Results

**Without pretraining**

Text generated by model trained only on "Pan Tadeusz" for 5k steps

![results_no_pretraining](assets/results_no_pretraining.jpg)

**With pretraining**

TODO


## Training on Athena

Installing dependencies
```sh
module load Python/3.10.4 CUDA/12.1.1 cuDNN/8.8.0.121-CUDA-12.0.0
python -m venv $SCRATCH/gpt
source $SCRATCH/gpt/bin/activate
pip install -r requirements.txt
export LD_LIBRARY_PATH=`python -m site --user-site`/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```

Training
```sh
srun --time=2:00:00 --mem=32G --ntasks 1 --gres=gpu:1 --partition=plgrid-gpu-a100 --account=$PLG_ACCOUNT --pty /bin/bash
module load Python/3.10.4 CUDA/12.1.1 cuDNN/8.8.0.121-CUDA-12.0.0
export LD_LIBRARY_PATH=`python -m site --user-site`/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
source $SCRATCH/gpt/bin/activate
python train.py config/train_pan_tadeusz_char.py
```

Sampling
```sh
python sample.py --out_dir=out-pan-tadeusz-char
```