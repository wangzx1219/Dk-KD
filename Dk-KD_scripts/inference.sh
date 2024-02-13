SUBSET=test
DOMAIN=it

DATA_PATH=/path/to/data-bin/$DOMAIN
MODEL_PATH=/path/to/model


# inference
CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA_PATH \
    --gen-subset $SUBSET \
    --path $MODEL_PATH \
    --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 \
    --source-lang de --target-lang en \
    --batch-size  64 \
    --scoring sacrebleu \
    --tokenizer moses \
    --remove-bpe