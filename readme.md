## Setup
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install sentencepiece
pip install editdistance
pip install sacrebleu==1.5.1
pip install pandas
pip install scipy

cd path/to/fairseq-0.10.2
pip install --editable ./
```

## Dataset
- Pre-training Librispeech
  - Source: https://www.openslr.org/12
  - Preprocessing: path/to/fairseq-0.10.2/examples/speech_to_text/prep_librispeech_data.py
- Evaluate: SpeechOcean762
  - Source: https://github.com/jimbozhang/speechocean762
  - Preprocessing: path/to/fairseq-0.10.2/examples/speech_to_text/prep_ocean_data.py
- Convert to phoneme
  - to_phoneme.py
  - to_phoneme_map.py
```bash
python path/to/fairseq-0.10.2/examples/speech_to_text/prep_librispeech_data.py \
    --output-root path/to/outputpath --vocab-type unigram \ 
    --vocab-size 10000
```

## Scripts
### Pre-training
#### word-level
```bash
LS_ROOT=path/to/librispeech
SAVE_DIR=path/to/checkpoint

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 path/to/fairseq-0.10.2/fairseq_cli/train.py ${LS_ROOT} \
  --train-subset train \
  --valid-subset valid \
  --distributed-backend 'nccl' --ddp-backend "no_c10d" \
  --save-dir ${SAVE_DIR} \
  --num-workers 4 \
  --max-tokens 50000 \
  --task nat_speech_to_text \
  --noise random_mask \
  --criterion nat_loss \
  --max-update 330 \
  --arch cmlm_s2t_transformer \
  --optimizer adam \
  --lr 5e-4 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 33 \
  --clip-norm 10.0 \
  --seed 1 \
  --update-freq 8 \
  --skip-invalid-size-inputs-valid-test \
```
#### phoneme-level
```bash
LS_ROOT=path/to/librispeech/phoneme
SAVE_DIR=path/to/checkpoint

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 path/to/fairseq-0.10.2/fairseq_cli/train.py ${LS_ROOT} \
  --train-subset train \
  --valid-subset valid \
  --distributed-backend 'nccl' --ddp-backend "no_c10d" \
  --save-dir ${SAVE_DIR} \
  --num-workers 4 \
  --max-tokens 50000 \
  --task nat_speech_to_text \
  --noise random_mask \
  --criterion nat_loss \
  --max-update 330 \
  --arch cmlm_s2t_transformer \
  --optimizer adam \
  --lr 5e-4 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 33 \
  --clip-norm 10.0 \
  --seed 1 \
  --update-freq 8 \
  --skip-invalid-size-inputs-valid-test \
  --no_bpe \
  --phoneme \
```

### Fine-tuning
#### word-level
```bash
LS_ROOT=path/to/speechocean762
SAVE_DIR=path/to/checkpoint

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 path/to/fairseq-0.10.2/fairseq_cli/train.py ${LS_ROOT} \
  --train-subset train \
  --valid-subset valid \
  --distributed-backend 'nccl' --ddp-backend "no_c10d" \
  --save-dir ${SAVE_DIR} \
  --num-workers 4 \
  --max-tokens 50000 \
  --task nat_speech_to_text \
  --noise no_noise \
  --criterion mse \
  --max-update 330 \
  --arch score_cmlm_s2t_transformer \
  --optimizer adam \
  --lr 5e-4 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 33 \
  --clip-norm 10.0 \
  --seed 1 \
  --update-freq 8 \
  --skip-invalid-size-inputs-valid-test \
  --load-pretrained-decoder-from path/to/Pre-training-checkpoint/checkpoint_best.pt \
  --load-pretrained-encoder-from path/to/Pre-training-checkpoint/checkpoint_best.pt \
```
#### phoneme-level
```bash
LS_ROOT=path/to/speechocean762/phoneme
SAVE_DIR=path/to/checkpoint

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 path/to/fairseq-0.10.2/fairseq_cli/train.py ${LS_ROOT} \
  --train-subset train \
  --valid-subset valid \
  --distributed-backend 'nccl' --ddp-backend "no_c10d" \
  --save-dir ${SAVE_DIR} \
  --num-workers 4 \
  --max-tokens 50000 \
  --task nat_speech_to_text \
  --noise no_noise \
  --criterion mse \
  --max-update 330 \
  --arch score_cmlm_s2t_transformer \
  --optimizer adam \
  --lr 5e-4 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 33 \
  --clip-norm 10.0 \
  --seed 1 \
  --update-freq 8 \
  --skip-invalid-size-inputs-valid-test \
  --no_bpe \
  --phoneme \
  --load-pretrained-decoder-from path/to/Pre-training-checkpoint/checkpoint_best.pt \
  --load-pretrained-encoder-from path/to/Pre-training-checkpoint/checkpoint_best.pt \
```

### Generate
#### word-level
```bash
SUBSET=test
LS_ROOT=path/to/dataset
SAVE_DIR=path/to/checkpoint

fairseq-generate ${LS_ROOT} --gen-subset ${SUBSET} --task nat_speech_to_text \
        --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} --batch-size 1 --beam 1 --scoring wer \
        --iter_decode_max_iter 0 --no_bpe --assessment --mask_length 1 \
```

#### phoneme-level
```bash
SUBSET=test
LS_ROOT=path/to/dataset
SAVE_DIR=path/to/checkpoint

fairseq-generate ${LS_ROOT} --gen-subset ${SUBSET} --task nat_speech_to_text \
        --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} --batch-size 1 --beam 1 --scoring wer \
        --iter_decode_max_iter 0 --no_bpe --assessment --mask_length 1 --phoneme \
```