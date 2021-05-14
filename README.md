# Incidental Lung Nodule Project Documentation
## Nodule Classification

### Train on LUNA16 

#### Config
Set `DATA_DIR` in config file.
- `DATA_DIR`: [***str***] Root directory of the dataset
- `CUBE_SIZE`: [***int***] Cube size to crop
- `LOAD_CLINICAL`: [***bool***] Whether to load clinical information
- `FLIP`: [***bool***] Whether to flip the image
- `ROTATE`: [***bool***] Whether to rotate the image
- `SWAP`: [***bool***] Whether to swap the axis

#### Train
```python classify.py -d=luna --gpu=0,1,2,3 --save_dir=results/luna_epoch50_sgd_lr0.001/ --train=True -b=16 -e=50 -lm=/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/classifier/results/luna_epoch50_sgd_lr0.001/res18```

#### Test
```python classify.py -d=luna --gpu=0,1,2,3 --save_dir=results/luna_epoch50_sgd_lr0.001/ --train=False -b=16 -e=50 -lm=/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/classifier/results/luna_epoch50_sgd_lr0.001/res18```

### Fine-tuning (LUNA16  --> Incidental )
#### Config
Set `DATA_DIR` in config file.
- `DATA_DIR`: [***str***] Root directory of the dataset
- `CUBE_SIZE`: [***int***] Cube size to crop
- `LOAD_CLINICAL`: [***bool***] Whether to load clinical information
- `FLIP`: [***bool***] Whether to flip the image
- `ROTATE`: [***bool***] Whether to rotate the image
- `SWAP`: [***bool***] Whether to swap the axis
#### Train
``` python classify.py -d=methodist --gpu=0,1,2,3 --save_dir=results/lunaInit_hasAug_multiNeg_adam0.001_e2028fold --train=True -b=16 -e=20 -lm=/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/classifier/results/luna_epoch50_adm_lr0.001/res18```
#### Test
``` python classify.py -d=methodist --gpu=0,1,2,3 --save_dir=results/lunaInit_hasAug_multiNeg_adam0.001_e2028fold --train=False -b=16 -e=20 -lm=/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/classifier/results/luna_epoch50_adm_lr0.001/res18```

#### K-Fold classification
1. set `SAVE_FOLDER`, `KFOLD` in `run_kfold.sh`
2. ``` bash run_kfold.sh```
- `KFOLD`: [***int***] K, number of folds (data split)
- `SAVE_FOLDER`: [***int***] Directary to save the results.
