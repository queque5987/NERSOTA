# 트위그팜 SOL Project 3기<br><br>- 구어체 NER 인식률 향상 프로젝트 -

## How to use
### • Pretrain
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
### • Finetuning
#### NERSOTA-BERT
```Python
python finetuning_bert.py -c ./NERSOTA_BERT ##--model_name NERSOTA_BERT --tokenizer "beomi/kcbert-base" --epochs 3 --batch_size 32
                           ##--learning_rate 1e-5 --corpus_name ner --max_length 64 --seed 7
```
#### NERSOTA-RoBERTa-t
```Python
python pretrain_roberta.py -c ./NERSOTA_RoBERTa_t --tokenizer "NERSOTA" ##--model_name NERSOTA_RoBERTa_u --epochs 3 --batch_size 32
                           ##--learning_rate 1e-5 --corpus_name ner --max_length 64 --seed 7
```
#### NERSOTA-RoBERTa-u
```Python
python pretrain_roberta.py -c ./NERSOTA_RoBERTa_u ##--model_name NERSOTA_RoBERT_u --tokenizer "BM-K/KoSimCSE-roberta" --epochs 3
                           ##--batch_size 32 --learning_rate 1e-5 --corpus_name ner --max_length 64 --seed 7
```
### • Inference
#### Text
```Python
python inference.py -m BERT -c ./NERSOTA_BERT/epoch=1-val_loss=0.18.ckpt --text 최예나는 24살이고, 대한민국의 가수야
                    ##-d NERSOTA_RoBERTa -t beomi/kcbert-base --max_length 128
```
#### Read as json file(will be saved *.ckpt.json)
```Python
python inference.py -m BERT -c ./NERSOTA_BERT/epoch=1-val_loss=0.18.ckpt -l test.json -s ./output
                    ##-d NERSOTA_RoBERTa -t beomi/kcbert-base --max_length 128
```
### • Output
```Python
$ python inference.py -m BERT -c ./trained/bert_base_t_kcbert_15/nlpbook/checkpoint-ner/epoch=4-val_loss=0.18.ckpt
{'output': ['B-PER',
            'I-PER',
            'I-PER',
            'I-PER',
            'O',
            'B-QTT',
            'I-QTT',
            'I-QTT',
            'I-QTT',
            'O',
            'O',
            'O',
            'B-LOC',
            'I-LOC',
            'I-LOC',
            'I-LOC',
            'I-LOC',
            'O',
            'O',
            'O',
            'O'],
 'output_b': '<PER>최예나는</PER> <QTT>24살이</QTT>고, <LOC>대한민국의</LOC> 가수야',
 'result': [{'predicted_tag': 'B-PER', 'token': '최', 'top_prob': '0.9664'},
            {'predicted_tag': 'I-PER', 'token': '##예', 'top_prob': '0.9739'},       
            {'predicted_tag': 'I-PER', 'token': '##나는', 'top_prob': '0.9906'},     
            {'predicted_tag': 'B-QTT', 'token': '24', 'top_prob': '0.6006'},
            {'predicted_tag': 'I-QTT', 'token': '##살이', 'top_prob': '0.8062'},
            {'predicted_tag': 'O', 'token': '##고', 'top_prob': '0.9998'},
            {'predicted_tag': 'O', 'token': ',', 'top_prob': '0.9997'},
            {'predicted_tag': 'B-LOC', 'token': '대한민국의', 'top_prob': '0.7439'},
            {'predicted_tag': 'O', 'token': '가수', 'top_prob': '0.648'},
            {'predicted_tag': 'O', 'token': '##야', 'top_prob': '0.9992'}],
 'sentence': '최예나는 24살이고, 대한민국의 가수야'}
```
## Dependency (CUDA = 11.0 기준)
transformers == 4.10.0<br>
pytorch == 1.7.1<br>
torchvision == 0.8.2<br>
torchaudio == 0.7.2<br>
cudatoolkit == 11.0<br>
[ratsnlp](https://github.com/ratsgo/ratsnlp) == 1.0.52<br>
tqdm<br>
gdown<
pprint
