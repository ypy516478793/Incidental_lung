#!/usr/bin/env bash

cd ../classifier
export PYTHONPATH=../:$PYTHONPATH
echo python path is ${PYTHONPATH}

SAVE_FOLDER=lunaInit_hasAug_testtesttest
KFOLD=2
GPU=0,1,2,3,4,5,6,7

for ((splitId=0; splitId<2; splitId+=1))
do
    echo "Start to run kfold splitId ${splitId}"
    python classify.py -d=methodist --gpu=${GPU} --kfold=${KFOLD} --splitId=${splitId} --save_dir=results/${SAVE_FOLDER}/${KFOLD}fold --train=False -b=16 -e=10 \
        -lm=/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/classifier/results/luna_test/res18/epoch_5.pt
done

python evaluate_kfold.py --kfold=${KFOLD} --save_dir=results_${KFOLD}fold
