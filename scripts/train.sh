#!/bin/bash

mkdir -p logs
export CUDA_VISIBLE_DEVICES=0,1,2,3

PATH_TO_DATA="data/"

MODEL_TYPE=${1}  # bert
MODEL_SIZE=${2}  # base
DATASET=${3}     # msmarco or asnq
ROUTINE=${4}     # all
SEED=42
LOG_ID=0

MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}-uncased

if [[ $DATASET = msmarco ]]
then
  EPOCHS=4
  LR=3e-6
fi

if [[ $DATASET = asnq ]]
then
  EPOCHS=2
  LR=2e-5
fi


echo ${MODEL_TYPE}-${MODEL_SIZE}/$DATASET $ROUTINE
python -um examples.run_highway_glue \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_NAME \
  --task_name $DATASET \
  --do_train \
  --do_lower_case \
  --data_dir $PATH_TO_DATA/$DATASET \
  --max_seq_length 512 \
  --per_gpu_eval_batch_size=1 \
  --per_gpu_train_batch_size=15 \
  --learning_rate $LR \
  --weight_decay 0.01 \
  --num_train_epochs $EPOCHS \
  --overwrite_output_dir \
  --seed $SEED \
  --output_dir ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/${ROUTINE}-${SEED} \
  --plot_data_dir ./plotting/ \
  --save_steps 0 \
  --train_routine $ROUTINE \
  --log_id $LOG_ID
