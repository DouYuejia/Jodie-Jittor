#!/bin/bash
export HOME=/data00/zch/

CUDA_VISIBLE_DEVICES=0 python3 jodie.py --network lastfm --model jodie --epochs 50 &
CUDA_VISIBLE_DEVICES=0 python3 jodie.py --network wikipedia   --model jodie --epochs 50 &
CUDA_VISIBLE_DEVICES=1 python3 jodie.py --network reddit --model jodie --epochs 50 &
CUDA_VISIBLE_DEVICES=1 python3 jodie.py --network mooc   --model jodie --epochs 50 &
wait