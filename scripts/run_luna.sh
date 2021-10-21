#!/usr/bin/env bash

cd ../classifier
export PYTHONPATH=../:$PYTHONPATH
echo python path is ${PYTHONPATH}

GPU=0,1,2,3,4,5,6,7

python classify.py --gpu=${GPU} -d luna --save_dir results/release_luna/ --train True -b 16

#for ((splitId=0; splitId<${KFOLD}; splitId+=1))
#do
#    echo "Start to run kfold splitId ${splitId}"
#    python classify.py --gpu=${GPU} --save_dir=results/luna/ --train=True -b=16
##    python train_vgg.py --gpu=${GPU} --kfold=5 --splitId=${splitId} --save_dir=AACR_results --train=False \
##        --load_model=AACR_results/bs_16.lr_0.001.beta_0.001.aug.balanceAfterSplit.kfold${splitId}
#done

#python evaluate_kfold.py --kfold=${KFOLD} --save_dir=results_${KFOLD}fold