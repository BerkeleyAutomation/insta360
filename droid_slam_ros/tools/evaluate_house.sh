#!/bin/bash


ETH_PATH=datasets/ETH3D-SLAM/training

# all "non-dark" training scenes
evalset=(
    sfm_house_loop
)

for seq in ${evalset[@]}; do
    python evaluation_scripts/test_eth3d.py --depth --datapath=$ETH_PATH/$seq --weights=droid.pth $@
done




