DOMAIN=it

DATA_PATH=/path/to/data-bin/$DOMAIN
PRETRAIN_MODEL_PATH=/path/to/wmt19.de-en.ffn8192.pt


CUDA_VISIBLE_DEVICES=1 python fairseq_cli/train.py \
    $DATA_PATH \
    --save-dir step1-$DOMAIN-1e4-eff0.5 \
    --source-lang de --target-lang en \
    --finetune-from-model $PRETRAIN_MODEL_PATH \
    --arch transformer_wmt_en_de_big --share-all-embeddings --encoder-ffn-embed-dim 8192 \
    --dropout 0.1 --weight-decay 0.0001 \
    --valid-subset valid \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy-step1 --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "lenpen": 0.6, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses --max-epoch 100 \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --no-epoch-checkpoints --no-last-checkpoints \
    --fp16 --num-workers 0 