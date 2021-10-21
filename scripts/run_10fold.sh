#!/usr/bin/env bash

cd ..
export PYTHONPATH=../:$PYTHONPATH
echo python path is ${PYTHONPATH}

KFOLD=10
GPU=0,1,2,3

for ((splitId=0; splitId<${KFOLD}; splitId+=1))
do
    echo "Start to run kfold splitId ${splitId}"
    python classifier/classify.py --gpu=${GPU} --kfold=${KFOLD} --splitId=${splitId} \
      --save_dir=classifier/results/LearnCurve_methodist/${KFOLD}fold --train=True -ts=0.1
done
