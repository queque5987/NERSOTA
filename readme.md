# 트위그팜 SOL Project 3기<br>'구어체 NER 인식률 향상 연구 프로젝트'
## 0. 개론
>최근 K-콘텐츠의 흥행으로 인한 글로벌 OTT서비스가 각광받기 시작하였고,<br>
>그로 인하여 기계번역 기술의 필요성 증대되었다.<br><br>
>이에 대해서 기계 번역의 품질 향상을 위한 연구를 계획하였으며<br>
>기계 번역의 subtask중 하나인 NER의 인식률 향상을 목표로 연구를 진행하였다.<br><br>
>한국어 구어체 NER인식률에 대한 선행 연구는 많이 진행되지 않았으며,<br>
>관련 모델과 평가 지표 또한 없었기 때문에 해당 연구의 기반을 닦아 K-콘텐츠의 기계 번역 품질을 향상시키고,<br>
>더불어 국내 OTT시장 확대에도 기여하고자 하였다.<br><br>
>해당 연구의 과제는 크게 두 가지로 나뉜다.<br>
>첫째, 현재 한국어 NER 모델을 구어체에 적용하였을 때의 인식률과 인식률이 낮게 나온다면 그러한 이유,<br>
>둘째, 현재 한국어 NER 모델을 구어체에서도 높은 인식률을 달성하려면 무엇이 필요한가.<br><br>
>본 연구는 두 번째 연구 주제에 집중하여 진행되었다.

***

## 1. Dataset
>구어체 NER 성능 비교 및 모델 학습을 위하여 한국어 구어체 NER Dataset을 구축하였다.<br><br>
>AIHub와 모두의 말뭉치에서 구어체 Corpus를 수집하였고,<br>
>NER 태깅이 되어 있는 Corpus를 구분하여 각 Pretrain Dataset, Finetuning Dataset에적용하였다.<br><br>
>모든 데이터셋은 8:1:1 로 Train, evaluation, test 데이터셋으로 분리하였다.
***

### • Pretrain Dataset
|Dataset|Domain|#Sentence|
|:------|:-----|:--------|
|[[AIHub](https://www.aihub.or.kr/)] 방송 콘텐츠 대본 요약 데이터|가족관련방송, 현대드라마, 역사극, 시사, 교양지식, 예능|897,113|
|[[AIHub](https://www.aihub.or.kr/)] 일상생활 및 구어체 한-중, 한-일 번역 병렬 말뭉치 데이터	|일상생활, 해외영업, 채팅	|1,313,479|
|[[AIHub](https://www.aihub.or.kr/)] 다국어 구어체 번역 병렬 말뭉치 데이터	|일상생활, 해외영업, 채팅	|1,298,467|
|[[모두의 말뭉치](https://corpus.korean.go.kr/)] 구어체 말뭉치	|공적 독백, 공적 대화, 준구어-대본	|8,843,306|
|[[모두의 말뭉치](https://corpus.korean.go.kr/)] 일상 대화 말뭉치 2020	|일상 대화, 협력적 대화	|141,721|

### • Finetuning Dataset
|Dataset|Domain|#Sentence|
|:------|:-----|:--------|
|[[AIHub](https://www.aihub.or.kr/)] 방송 콘텐츠 한-중, 한-일 번역 병렬 말뭉치 데이터	|TV 방송, 라디오 방송	|1,350,000|
|[[AIHub](https://www.aihub.or.kr/)] 일상생활 및 구어체 한-영 번역 병렬 말뭉치 데이터	|일상생활, 해외영업, 해외고객과의 채팅	|1,350,000|
|[[모두의 말뭉치](https://corpus.korean.go.kr/)] 개체명 분석 말뭉치 2021	|공적 독백, 공적 대화, 준구어-대본	|552,293|

**NER 태깅이 된 문장만 활용함*

#### - Dataset A (Finetuning Dataset)
>AIHub데이터셋은 PER, ORG, WORK_OF_ART, PRODUCT 4가지의 태그로 태깅이 되어 있고,<br>
>모두의 말뭉치 데이터셋은 PERSON, CIVILIZATION, QUANTITY, LOCATION ,ORGANIZATION, <br>
>DATE, TIME, ARTIFACTS, ANIMAL, TIME, EVENT, PLANT, MATERIAL, THEORY, STUDY_FIELD<br>
>15가지의 태그로 태깅이 되어 있는데 이는 TTA기준을 따르는 태깅 기준이다.<br><br>
>Dataset A(4)는 AIHub 태깅 기준에 맞추어 전체 태그를 4개로 축소한 데이터셋이다.<br>
>PERSON, ORGANIZATION를 제외한 태그는 제거하였고, ARTIFACTS는 세부 태깅으로 구분하여 WORK_OF_ART와 PRODUCT로 구분하였다.<br><br>
>Dataset A(15)는 모두의 말뭉치, TTA 기준을 적용하여 전체 태그를 15가지로 확대한 데이터셋이다.<br>
>WORK_OF_ART와 PRODUCT는 ARTIFACTS에 포함하였다. 중복된 문장 68,991문장이 검출되었는데<br>
>이는 모두의 말뭉치 데이터셋, 즉 15가지 태깅이 되어 있는 문장으로 대체하였다.<br><br>
>하지만, AIHub에서 가져온 데이터가 77,333문장이기 때문에<br>
>최소 8,342문장이 4개 태그 기준으로 태깅이 되어 있음을 알 수 있다.
***

#### - Dataset B (Finetuning Dataset + NERSOTA Dataset)
>기존의 0.73의 F1 Score는 Dataset A(15)의 수량이 부족하고<br>
>최소 8,342문장이 4개의 태그 기준으로 태깅이 되어 있어 데이터셋의 신뢰도가 부족하기 때문이라고 판단하였다.<br><br>
>이에 데이터를 추가 구축하고자 하였다.<br>
>총 14명의 작업자에게 TTA기준 작업 가이드라인을 배포하고 작업을 진행하였다.<br><br>
>작업에는 LabelStudio를 사용하였다.<br>
>작업 초반, 모든 작업을 평가하고 피드백을 주어 작업물의 완성도를 높였으며 Google chat을 활용하여 실시간 질의응답 또한 진행하였다.<br>
>또한, 작업자 개인의 작업 능률과 이해도에 따라 작업량을 조절하며 진행하였다.<br><br>
>작업완료 후, json파일을 검토하였으며 범위가 지정되어 있지 않거나,<br>
>태그가 누락된 문장, 태깅 대상이 누락된 문장 등 5000여 문장을 수정하였다.<br>
***
#### - Dataset C (개체명 분석 말뭉치 2021)
>Dataset C는 Finetuning Dataset의 개체명 분석 말뭉치 2021 데이터만을 사용하였다.
***

## 2. 모델 연구
>NER에서 F1 measure를 적용시키기 위하여 정답 label을 Sequence label 형태로 변환하여 측정하였는데,<br>
>변환된 Sequence label은 O 태그의 개수가 현저히 많아 O태그에 대한 가중치가 높게 측정되어<br>
>F1 Score 또한 비정상적으로 높게 측정되는 현상이 있었다.<br><br>
>이로 인하여 각 태그의 개수에 상관없이 가중치를 동등하게 부여하는 Macro F1 Score(이하 F1 Score)를 선택하여 비교하였다.

|NER Model           |Dataset A(n)|
|:-------------------|:----------:|
|[KcBERT](https://github.com/Beomi/KcBERT)(Naver NER)   |0.21        |
|[LETR API](https://www.letr.ai/)            |0.50        |
|[KoBERT](https://github.com/SKTBrain/KoBERT)              |0.34        |
|[tf-xlm-r-ner-40-lang](https://huggingface.co/jplu/tf-xlm-r-ner-40-lang)|0.25        |
|[KcBERT](https://github.com/Beomi/KcBERT)(Finetuned)   |**0.73**    |
>기존 문어체로 학습된 모델의 Macro averaged F1 Score(*이하 F1 Score*)를 측정한 결과 매우 낮은 수준으로 측정되었다.<br><br>
>또한, 기존 LM을 Finetuning한 것만으로도 F1 Score 0.21 >> 0.73으로 유의미한 성능 향상을 보였다.

>다음으로, LM의 특화 정도에 따른 성능 변화를 관찰하기 위하여 구어체 특화 PLM을 학습하였다.
![image](https://user-images.githubusercontent.com/64758868/209208480-430bee22-1a87-4627-8817-c9183fa4c531.png)
![image](https://user-images.githubusercontent.com/64758868/209208568-7f0e0d5e-c162-42e6-9170-d1576fa1925e.png)
![image](https://user-images.githubusercontent.com/64758868/209208575-e60be666-ce1a-456f-909d-acb74bbdd380.png)

|PLM                |Dataset A(4)|Dataset A(15)|Dataset B |Dataset C |
|:------------------|:----------:|:------------:|:--------:|:--------:|
|[KcBERT](https://github.com/Beomi/KcBERT)            |0.73        |0.73|**0.78**  |0.84  |
|NERSOTA-BERT       |**0.76**    |-|**0.78**  |0.82      |
|NERSOTA-RoBERTa-t  |*0.38*      |-|*0.24*    |*0.33*    |
|NERSOTA-RoBERTa-u  |*0.40*      |-|*0.29*    |-         |
|NERSOTA-Electra    |-           |-|-         |**0.92**  |
>Pretrain Dataset을 사용하여 BERT-base, RoBERTa-base의 학습을 진행하여 PLM을 구축하였다.<br><br>
>[KcBERT](https://github.com/Beomi/KcBERT)는 기존 LM을 그대로 사용하여 baseline으로 사용하였다..<br>
>NERSOTA-BERT에는 [KcBERT](https://github.com/Beomi/KcBERT) Tokenizer를 사용하였고,<br>
>NERSOTA-RoBERTa-t는 Pretrain Dataset으로 학습된 Tokenizer,<br>
>NERSOTA-RoBERTa-u는 [KoSimSCE-roberta](https://github.com/BM-K/Sentence-Embedding-is-all-you-need) Tokenizer 두 종류로 진행하였다.<br><br>
>PLM을 각각의 Dataset으로 Finetuning하여 그 결과의 F1 Score를 측정한 값이다.<br>

## 3. 결론
>Dataset A(4)에서 같은 BERT-base 기반 LM KcBERT에서 NERSOTA-BERT-base로 교체한 것 만으로 성능 향상을 보았다.<br>
>이로써 구어체 특화 PLM을 사용하는 것만으로도 유의미한 성능 향상을 이룩할 수 있다는 점을 증명하였다.<br><br>
>하지만, Dataset B에서 동일하게 LM을 교체하였으나, 점수가 동일하게 나왔다는 점<br>
>그리고 Dataset C에서 오히려 NERSOTA-BERT-base가 떨어지는 점을 보았을 때<br>
>15가지의 태그로 진행할 때는 단순 PLM 교체로는 부족하다는 점을 알 수 있다.<br><br>
>NERSOTA-BERT-base는 MLM만을 사용하여 학습되었으나, KcBERT의 경우는 MLM, NSP를 진행하여 학습되어 F1 Score가 높게 측정되었다.<br>
>NERSOTA-Electra-small의 경우는 RTD를 사용하여 학습된 차이점이 존재한다.<br>
>이러한 모델 구조상의 차이로 인하여 NERSOTA-Electra의 경우 높은 F1 Score를 달성하였다.<br><br>
>NERSOTA-RoBERTa-base 모델을 사용하여 Tokenizer의 학습의 유의미함을 연구하고자 하였으나,<br>
>전체 RoBERTa-base기반 PLM의 F1 Score가 비정상적으로 낮게 측정되었고,<br>
>구어체로 학습된 Tokenizer가 오히려 성능을 하락하였다.<br>
>이에 대해서 Tokenizer의 학습 유무는 구어체 NER 인식률에서 성능 차이를 보이지 않았다고 볼 수 있으나, 추가 연구를 진행해야 한다.

## 4. 한계점
>PLM의 학습 방법에서 DatasetB와 모두의 말뭉치 Test dataset에서 비슷하거나 오히려 낮은 점수를 기록한 이유를 찾았으나,<br>
>문장 내에서 Classification을 진행하는 NER작업에서 문장 내의 문맥을 파악하는 MLM만을 진행하는 것이 부족한지는 의문이다.<br>
>이를 해결하기 위해선 우선 Pretrain Dataset의 양을 증강할 필요가 있다.<br>
>또한, KcBERT는 약 9천만 문장이 사용되어 데이터의 학습량에 따른 변인도 제한하여 진행해야 했다.<br><br>
>NSP를 추가로 진행한 PLM, NSP만 진행한 PLM을 구축하여 비교하면<br>PLM의 MLM학습 방법이 부족한 방법인가에 대한 명제를 증명할 수 있을 것이다.<br><br>
>그리고, RoBERTa-base기반의 PLM이 전체적으로 낮은 성능을 기록하여 모델 구조에 따른 차이점을 연구하는 데 실패하였다.<br>
>RoBERTa의 모델 구조가 구어체 NER에 있어서 낮은 성능을 보이는 것일 가능성도 존재하나 확률은 희박하고,<br>
>학습 코드를 재검토 후 수정할 필요성이 존재한다.


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

***
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
***
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
***
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
***
## Dependency (CUDA = 11.0 기준)
transformers == 4.10.0<br>
pytorch == 1.7.1<br>
torchvision == 0.8.2<br>
torchaudio == 0.7.2<br>
cudatoolkit == 11.0<br>
[ratsnlp](https://github.com/ratsgo/ratsnlp) == 1.0.52<br>
tqdm<br>
gdown<br>
pprint

***
## Reference
[BERT](https://github.com/google-research/bert)<br>
[RoBERTa](https://huggingface.co/roberta-base)<br>
[KcBERT](https://github.com/Beomi/KcBERT)<br>
[tf-xlm-r-ner-40-lang](https://huggingface.co/jplu/tf-xlm-r-ner-40-lang)<br>
[KoBERT](https://github.com/SKTBrain/KoBERT)<br>
[LETR API](https://www.letr.ai/)<br>
[KoSimSCE-roberta](https://github.com/BM-K/Sentence-Embedding-is-all-you-need)<br>


[AIHub](https://www.aihub.or.kr/)<br>
[모두의 말뭉치](https://corpus.korean.go.kr/)<br>

[트위그팜](https://www.twigfarm.net/)<br>
[Label Studio](https://labelstud.io/)
