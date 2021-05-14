#!/usr/bin/env bash

cd ../classifier
export PYTHONPATH=../:$PYTHONPATH
echo python path is ${PYTHONPATH}

SAVE_FOLDER=hasAug
KFOLD=28
GPU=0,1,2,3,4,5,6,7

for ((splitId=7; splitId<14; splitId+=1))
do
    echo "Start to run kfold splitId ${splitId}"
    python classify_aug.py --gpu=${GPU} --kfold=${KFOLD} --splitId=${splitId} --save_dir=results/${SAVE_FOLDER}/${KFOLD}fold --train=True -b=16
#    python train_vgg.py --gpu=${GPU} --kfold=5 --splitId=${splitId} --save_dir=AACR_results --train=False \
#        --load_model=AACR_results/bs_16.lr_0.001.beta_0.001.aug.balanceAfterSplit.kfold${splitId}
done

#python evaluate_kfold.py --kfold=${KFOLD} --save_dir=results_${KFOLD}fold