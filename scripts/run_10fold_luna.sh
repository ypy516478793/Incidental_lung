#!/usr/bin/env bash

cd ..
export PYTHONPATH=../:$PYTHONPATH
echo python path is ${PYTHONPATH}

KFOLD=10
GPU=0,1,2,3

# Training and testing
for ts in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    echo "test size is " ${ts}
    for ((splitId=0; splitId<${KFOLD}; splitId+=1))
    do
        echo "Start to run kfold splitId ${splitId}"
        python classifier/classify.py -d=luna --gpu=${GPU} --kfold=${KFOLD} --splitId=${splitId} \
          --save_dir=classifier/results/LearnCurve_luna/${ts}testSize_${KFOLD}fold --train=True -ts=${ts}
    done
done


# Evaluation
for ts in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    echo "test size is " ${ts}
    python classifier/evaluate_kfold.py --kfold=${KFOLD} \
        --save_dir=classifier/results/LearnCurve_luna/${ts}testSize_${KFOLD}fold
done