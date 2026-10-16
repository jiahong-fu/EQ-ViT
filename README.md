# EQ-ViT

Implementation of **Vanilla Group Equivariant Vision Transformer**, including equivariant Swin Transformer and Vision Transformer models for ImageNet-1K.

| Directory | Contents | Main EQ implementation |
| --- | --- | --- |
| [`swin/`](swin/) | Swin baselines, EQ-Swin, configurations, training and evaluation | [`swin_transformer_eq_pos_eq_patchmerging_conv.py`](swin/models/swin_transformer_eq_pos_eq_patchmerging_conv.py) |
| [`vit/`](vit/) | Modified `timm` package, ViT baselines, EQ-ViT, training and evaluation | [`vision_transformer_eq_norm_eq.py`](vit/timm/models/vision_transformer_eq_norm_eq.py) |

The Swin reproduction below uses **Swin-Tiny-EQ with embedding dimension 192**. Swin-Small and Swin-Base experiments were not run due to compute limits. The ViT command trains `vit_small_eq_patch16_224`; Tiny and Base EQ model definitions are also included.

## Data and checkpoints

Prepare ImageNet-1K using the standard class-folder layout:

```text
ImageNet1K/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```

Dataset images, trained checkpoints, and training outputs are not included. Set `DATA_PATH` to your dataset directory. The Swin loader currently requires both `train/` and `val/`, including during evaluation. To reproduce the supplied Swin run, obtain `ckpt_epoch_298.pth` separately and set `CHECKPOINT` to its location; no checkpoint download is provided in this repository.

```bash
git clone https://github.com/jiahong-fu/EQ-ViT.git
cd EQ-ViT
```

## Swin-Tiny-EQ evaluation

Use a separate environment for Swin, which depends on `timm==0.4.12`. The commands below use Python 3.9, PyTorch 2.5.1, torchvision 0.20.1, and CUDA 12.1 wheels.

```bash
conda create -n eq-swin python=3.9 -y
conda activate eq-swin
cd swin
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

DATA_PATH=/path/to/ImageNet1K
CHECKPOINT=/path/to/ckpt_epoch_298.pth

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port 12350 \
    main.py --eval \
    --cfg configs/swin/swin_tiny_eq_pos_eq_patch4_window7_224.yaml \
    --resume "$CHECKPOINT" \
    --data-path "$DATA_PATH" \
    --batch-size 128 \
    --output output/swin_eq_pos_eq_eval_repro
```

The original checkpoint location, relative to `swin/`, is `output/swin_tiny_eq_pos_eq_patch4_window7_224_192embed_dim_patchmerging_conv/default/ckpt_epoch_298.pth`. Training and baseline options are in [`swin/get_started.md`](swin/get_started.md).

## ViT-Small-EQ training

Start from the repository root in a new shell. Use a separate environment and install the **local modified `timm` package**, which registers the EQ models. The commands below use Python 3.10, PyTorch 2.4.0, torchvision 0.19.0, and CUDA 12.1 wheels.

```bash
conda create -n eq-vit python=3.10 -y
conda activate eq-vit
cd vit
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e .

DATA_PATH=/path/to/ImageNet1K

CUDA_VISIBLE_DEVICES=0,1 ./distributed_train.sh 2 18945 \
    --data-dir "$DATA_PATH" \
    --model vit_small_eq_patch16_224 \
    --sched cosine --epochs 300 --opt adamw -j 8 \
    --warmup-lr 1e-6 --mixup .2 \
    --model-ema --model-ema-decay 0.99996 \
    --aa rand-m9-mstd0.5-inc1 --remode pixel --reprob 0.25 \
    --amp --lr 8e-4 --weight-decay .05 \
    --drop 0.1 --drop-path .1 -b 288
```

This uses two GPUs with batch size 288 per GPU. Training outputs are written under `vit/output/train/`. `vit_tiny_eq_patch16_224`, `vit_small_eq_patch16_224`, and `vit_base_eq_patch16_224` are registered from `vision_transformer_eq_norm_eq.py`, preserving the effective model selection in the original code. The alternative `vision_transformer_eq.py` source is retained for reference.

## Acknowledgements

This repository builds on [Microsoft Swin Transformer](https://github.com/microsoft/Swin-Transformer) and [Hugging Face PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models). Upstream documentation and copyright notices are retained. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), [`swin/LICENSE`](swin/LICENSE), and [`vit/LICENSE`](vit/LICENSE) for attribution and upstream license texts.
