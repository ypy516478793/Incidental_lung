#!/usr/bin/env bash

for ((splitId=0; splitId<5; splitId+=1))
do
    echo "process $i epoch"
    echo "Start to run kfold splitId ${splitId}"
    python train_vgg.py --gpu=7 --kfold=5 --splitId=${splitId} --save_dir=AACR_results --train=True
    python train_vgg.py --gpu=7 --kfold=5 --splitId=${splitId} --save_dir=AACR_results --train=False \
        --load_model=AACR_results/bs_16.lr_0.001.beta_0.001.aug.balanceAfterSplit.kfold${splitId}

done