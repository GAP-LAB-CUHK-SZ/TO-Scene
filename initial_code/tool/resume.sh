#!/bin/sh
##SBATCH -p gpu
#SBATCH -p q6,q8
#SBATCH --gres=gpu:4
#SBATCH -c 40

#SBATCH -x isl-gpu16,isl-gpu9,isl-gpu10,isl-gpu11,isl-gpu12


export PYTHONPATH=./
eval "$(conda shell.bash hook)"
PYTHON=python
conda activate pt170

TRAIN_CODE=train.py
TEST_CODE=test.py

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml


now=$(date +"%Y%m%d_%H%M%S")
$PYTHON ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  save_path ${exp_dir} \
  resume ${model_dir}/model_last.pth \
  2>&1 | tee ${exp_dir}/train-$now.log

sbatch --qos=high tool/test.sh $1 $2

: '
$PYTHON -u ${exp_dir}/${TEST_CODE} \
  --config=${config} \
  save_folder ${result_dir}/best \
  model_path ${model_dir}/model_best.pth \
  2>&1 | tee ${exp_dir}/test_best-$now.log

now=$(date +"%Y%m%d_%H%M%S")
$PYTHON -u ${exp_dir}/${TEST_CODE} \
  --config=${config} \
  save_folder ${result_dir}/last \
  model_path ${model_dir}/model_last.pth \
  2>&1 | tee ${exp_dir}/test_last-$now.log
'
