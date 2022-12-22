# 트위그팜 SOL Project 3기 "구어체 NER 인식률 향상" 프로젝트

## How to use
### Pretrain
#### NERSOTA-BERT
```Python
python pretrain_bert.py ##--model_name NERSOTA_BERT --epochs 20 --batch_size 32
                        ##--eval_steps 50000 --save_steps 300000 --max_length 64 --on_memory False
```
#### NERSOTA-RoBERTa-t
```Python
python pretrain_roberta.py -t True ##--model_name NERSOTA_RoBERTa_t --epochs 20 --batch_size 32
                           ##--eval_steps 50000 --save_steps 300000 --max_length 64
```
#### NERSOTA-RoBERTa-u
```Python
python pretrain_roberta.py ##-t False --model_name NERSOTA_RoBERTa_u --epochs 20 --batch_size 32
                           ##--eval_steps 50000 --save_steps 300000 --max_length 64
```
***
### Finetuning
#### NERSOTA-BERT
```Python
python finetuning_bert.py -c ./*.ckpt ##--model_name NERSOTA_BERT --tokenizer "beomi/kcbert-base" --epochs 3 --batch_size 32
                           ##--learning_rate 1e-5 --corpus_name ner --max_length 64 --seed 7
```
#### NERSOTA-RoBERTa-t
```Python
python pretrain_roberta.py -c ./*.ckpt --tokenizer "NERSOTA" ##--model_name NERSOTA_RoBERTa_u --epochs 3 --batch_size 32
                           ##--learning_rate 1e-5 --corpus_name ner --max_length 64 --seed 7
```
#### NERSOTA-RoBERTa-t
```Python
python pretrain_roberta.py -c ./*.ckpt ##--model_name NERSOTA_RoBERT_u --tokenizer "BM-K/KoSimCSE-roberta" --epochs 3 --batch_size 32
                           ##--learning_rate 1e-5 --corpus_name ner --max_length 64 --seed 7
```
***
### Inference
#### NERSOTA-BERT
```Python
afsdfasdfasd
```

## Dependency (CUDA = 11.0 기준)
transformers == 4.10.0<br>
pytorch == 1.7.1<br>
torchvision == 0.8.2<br>
torchaudio == 0.7.2<br>
cudatoolkit == 11.0<br>
[ratsnlp](https://github.com/ratsgo/ratsnlp) == 1.0.52<br>
tqdm<br>
gdown
