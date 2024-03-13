#!/bin/bash


# # Conda environment setup
# conda env create -f environment.yml
# conda init bash

CKPTS_DIR="./text2light_released_model"

# HDR
echo "Have you called \"conda activate text2light\"?"

while IFS= read -r line
do
  # Replace 'your_command' with your actual command
  echo -e "\nPromt << $line"
  python3 text2light.py \
    -rg $CKPTS_DIR/global_sampler_clip \
    -rl $CKPTS_DIR/local_sampler_outdoor \
    --outdir ./generated_panorama \
    --text "$line" \
    --clip clip_emb.npy \
    --sritmo $CKPTS_DIR/sritmo.pth \
    --sr_factor 1
done < "alt_outdoor.txt"
exit 0

# Other Parameters
# --sr_factor 4 # scale factor for super resolution
# --text "photorealistc baseball diamond" \


# Skip HDR
# python3 text2light.py \
#     -rg $CKPTS_DIR/global_sampler_clip \
#     -rl $CKPTS_DIR/local_sampler_outdoor \
#     --outdir ./generated_panorama \
#     --text "photorealistc baseball diamond" \
#     --clip clip_emb.npy \
#     --sr_factor 4
