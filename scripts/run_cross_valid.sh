#!/usr/bin/env bash

cd ../classifier

KFOLD=142
GPU=0

for ((splitId=0; splitId<${KFOLD}; splitId+=1))
do
    echo "Start to run kfold splitId ${splitId}"
    python classify.py --gpu=${GPU} --kfold=${KFOLD} --splitId=${splitId} --save_dir=results_${KFOLD}fold --train=True
#    python train_vgg.py --gpu=${GPU} --kfold=5 --splitId=${splitId} --save_dir=AACR_results --train=False \
#        --load_model=AACR_results/bs_16.lr_0.001.beta_0.001.aug.balanceAfterSplit.kfold${splitId}
done

python evaluate_kfold.py --kfold=${KFOLD} --save_dir=results_${KFOLD}fold