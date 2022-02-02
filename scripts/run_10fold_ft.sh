#!/usr/bin/env bash

cd ..
export PYTHONPATH=$(pwd)
echo python path is ${PYTHONPATH}

KFOLD=10
GPU=0,1,2,3

## 10 fold Training and testing
#for ((splitId=0; splitId<${KFOLD}; splitId+=1))
#do
#    echo "Start to run kfold splitId ${splitId}"
#    python classifier/classify.py --gpu ${GPU} -d methodist --kfold ${KFOLD} --splitId ${splitId} \
#        --save_dir classifier/results/release_methodist/min_max_load_luna_cube32/${KFOLD}fold --train True -b 16 -ts=0.1 \
#        -p ./Methodist_incidental/data_Ben/preprocessed/ -lm classifier/results/release_luna/res18/epoch_7.pt
#done
## Evaluation
#python classifier/evaluate_kfold.py --kfold=${KFOLD} \
#    --save_dir=classifier/results/release_methodist/min_max_load_luna_cube32/${KFOLD}fold


# 10 fold Training and testing new lung mask
for ((splitId=0; splitId<${KFOLD}; splitId+=1))
do
    echo "Start to run kfold splitId ${splitId}"
    python classifier/classify.py --gpu ${GPU} -d methBenMinmaxNew --kfold ${KFOLD} --splitId ${splitId} \
        --save_dir classifier/results/release_methodist/min_max_load_luna_newMask_cube32/${KFOLD}fold --train True -b 16 -ts=0.1 \
        -lm classifier/results/release_luna/res18/epoch_7.pt
done
# Evaluation
python classifier/evaluate_kfold.py --kfold=${KFOLD} \
    --save_dir=classifier/results/release_methodist/min_max_load_luna_newMask_cube32/${KFOLD}fold


### Learning curve by varying test size
## Training and testing
#for ts in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#do
#    echo "test size is " ${ts}
#    for ((splitId=0; splitId<${KFOLD}; splitId+=1))
#    do
#        echo "Start to run kfold splitId ${splitId}"
#        python classifier/classify.py --gpu=${GPU} --kfold=${KFOLD} --splitId=${splitId} \
#          --save_dir=classifier/results/LearnCurve_methodist/${ts}testSize_${KFOLD}fold --train=True -ts=${ts}
#    done
#done
#
## Evaluation
#for ts in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#do
#    echo "test size is " ${ts}
#    python classifier/evaluate_kfold.py --kfold=${KFOLD} \
#        --save_dir=classifier/results/LearnCurve_methodist/${ts}testSize_${KFOLD}fold
#done
