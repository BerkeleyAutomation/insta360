#!/bin/bash


ETH_PATH=datasets/Zed/training

# all "non-dark" training scenes
evalset=(
    #d455_kitchen_loop
    zed_kitchen_loop
)

for seq in ${evalset[@]}; do
    python evaluation_scripts/test_zed.py --depth --datapath=$ETH_PATH/$seq --weights=droid.pth $@
done




