#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA="data/"

MODEL_TYPE=${1}  # bert
MODEL_SIZE=${2}  # base
DATASET=${3}     # msmarco or asnq
ROUTINE=${4}     # all
PARTITION_LIST=${5}
SEED=42
LOG_ID=0

PARTITION_CACHE=$PATH_TO_DATA/$DATASET/partition_cache
EVAL_COL='dev_partitions'
if [[ $DATASET = msmarco ]]
then
  TARGET_MODEL=epoch-3
  EVAL_RESULT_DIR=evaluation/msmarco
fi

if [[ $DATASET = asnq ]]
then
  TARGET_MODEL=epoch-1
  EVAL_RESULT_DIR=evaluation/asnq
fi

mkdir -p $PARTITION_CACHE
ln -sf $PWD/saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/${ROUTINE}-${SEED}/vocab.txt \
      ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/${ROUTINE}-${SEED}/${TARGET_MODEL}


echo ${MODEL_TYPE}-${MODEL_SIZE}/$DATASET $ROUTINE
python -um examples.run_highway_glue \
  --model_type $MODEL_TYPE \
  --model_name_or_path ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/${ROUTINE}-${SEED}/${TARGET_MODEL} \
  --task_name $DATASET \
  --do_eval \
  --do_lower_case \
  --data_dir $PATH_TO_DATA/$DATASET \
  --evaluation_dir ${EVAL_RESULT_DIR} \
  --eval_collection_dir $PATH_TO_DATA/$DATASET/$EVAL_COL \
  --todo_partition_list ${PARTITION_LIST} \
  --output_dir ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/${ROUTINE}-${SEED}/${TARGET_MODEL} \
  --max_seq_length 512 \
  --seed $SEED \
  --eval_highway \
  --quick_eval \
  --per_gpu_eval_batch_size=256 \
  --train_routine $ROUTINE \
  --log_id $LOG_ID \
  --output_score_file
