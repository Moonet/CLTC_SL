export GLUE_DIR=./data

PYTHONPATH=../ CUDA_VISIBLE_DEVICES=0,1,2,3 python run_vat.py \
  --task_name mld \
  --do_train \
  --do_eval \
  --do_lower_case \
  --lang japanese \
  --data_dir $GLUE_DIR/MLDoc/ \
  --bert_model bert-base-multilingual-cased \
  --max_seq_length 128 \
  --train_batch_size 48 \
  --learning_rate 2e-5 \
  --num_k 40 \
  --num_self_train 1 \
  --num_train_epochs 10.0 \
  --output_dir /tmp/mld_ja_vat_output/ \