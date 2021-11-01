#!/usr/bin/env bash

cd ..
export PYTHONPATH=$(pwd)
echo python path is ${PYTHONPATH}

GPU=0,1,2,3,4,5,6,7

## LUNA test
#python classify.py --gpu=${GPU} -d luna --save_dir results/luna_test/res18/ --train False -b 16 \
#    -lm results/luna_test/res18/epoch_5.pt
#
#python evaluate_kfold.py --kfold=1 --save_dir=results/luna_test/res18/


# Methodist test (load LUNA model)
python classifier/classify.py --gpu=${GPU} -d methodist --save_dir classifier/results/release_methodist/min_max_cube32/ --train False -b 16 \
    -p ./Methodist_incidental/data_Ben/preprocessed/ -lm classifier/results/release_luna/res18/epoch_7.pt -la True

python classifier/evaluate_kfold.py --kfold=1 --save_dir=classifier/results/release_methodist/min_max_cube32/res18/



## Methodidst AACR test
#for ((splitId=0; splitId<${KFOLD}; splitId+=1))
#do
#    echo "Start to run kfold splitId ${splitId}"
#    python classify.py --gpu=${GPU} --save_dir=results/luna/ --train=True -b=16
##    python train_vgg.py --gpu=${GPU} --kfold=5 --splitId=${splitId} --save_dir=AACR_results --train=False \
##        --load_model=AACR_results/bs_16.lr_0.001.beta_0.001.aug.balanceAfterSplit.kfold${splitId}
#done

#python evaluate_kfold.py --kfold=${KFOLD} --save_dir=results_${KFOLD}fold