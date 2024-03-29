#!/bin/bash


ETH_PATH=datasets/D455/training

# all "non-dark" training scenes
evalset=(
    #d455_kitchen_loop
    d455_hallway
)

for seq in ${evalset[@]}; do
    python evaluation_scripts/test_d455.py --depth --datapath=$ETH_PATH/$seq --weights=droid.pth $@
done




