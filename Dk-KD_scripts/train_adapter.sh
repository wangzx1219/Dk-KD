DOMAIN=it
declare -A  FFN_SIZE
FFN_SIZE[medical]=8192; FFN_SIZE[law]=8192; FFN_SIZE[it]=8192; FFN_SIZE[koran]=512

DATA_PATH=/path/to/data-bin/$DOMAIN
PRETRAIN_MODEL_PATH=/path/to/wmt19.de-en.ffn8192.pt


CUDA_VISIBLE_DEVICES=2 python fairseq_cli/train.py \
    $DATA_PATH \
    --save-dir adapter-$DOMAIN-knf2-3e4-${FFN_SIZE[$DOMAIN]}-eff0.0-0.8 \
    --source-lang de --target-lang en \
    --finetune-from-model $PRETRAIN_MODEL_PATH \
    --arch transformer_wmt_en_de_big --share-all-embeddings --encoder-ffn-embed-dim 8192 \
    --dropout 0.1 --weight-decay 0.0001 \
    --valid-subset valid \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 3e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --criterion label_smoothed_cross_entropy-step2 --label-smoothing 0.1 \
    --max-tokens 8192 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "lenpen": 0.6, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses --max-epoch 80 \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --no-epoch-checkpoints --no-last-checkpoints \
    --fp16 --num-workers 0 \
    --encoder-append-adapter --decoder-append-adapter \
    --only-update-adapter --adapter-ffn-dim ${FFN_SIZE[$DOMAIN]} --activate-adapter