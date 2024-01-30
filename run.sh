#!/bin/bash


# # Conda environment setup
# conda env create -f environment.yml
# conda init bash

CKPTS_DIR="./text2light_released_model"

# python3 text2light.py \
#     -rg $CKPTS_DIR/global_sampler_clip \
#     -rl $CKPTS_DIR/local_sampler_outdoor \
#     --outdir ./generated_panorama \
#     --text "A storm is approaching on the horizon" \
#     --clip clip_emb.npy \
#     --sritmo $CKPTS_DIR/sritmo.pth \
#     --sr_factor 4


# Skip HDR
python3 text2light.py \
    -rg $CKPTS_DIR/global_sampler_clip \
    -rl $CKPTS_DIR/local_sampler_outdoor \
    --outdir ./generated_panorama \
    --text "A storm is approaching on the horizon" \
    --clip clip_emb.npy \
    --sr_factor 4
