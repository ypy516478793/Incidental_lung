#!/usr/bin/env bash

cd ../classifier
export PYTHONPATH=../:$PYTHONPATH
echo python path is ${PYTHONPATH}

SAVE_FOLDER=hasAug
KFOLD=28
GPU=0,1,2,3,4,5,6,7

for ((splitId=0; splitId<7; splitId+=1))
do
    echo "Start to run kfold splitId ${splitId}"
    python classify_aug.py --gpu=${GPU} --kfold=${KFOLD} --splitId=${splitId} --save_dir=results/${SAVE_FOLDER}/${KFOLD}fold --train=True -b=16
done

#python evaluate_kfold.py --kfold=${KFOLD} --save_dir=results_${KFOLD}fold