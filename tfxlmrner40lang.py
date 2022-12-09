from transformers import pipeline
# import spacy
# pip install --no-cache-dir transformers sentencepiece
# pip install tensorflow <- ?
# pip install torch 1.12.1
# https://aka.ms/vs/16/release/vc_redist.x64.exe

from transformers import AutoTokenizer, AutoModelForTokenClassification, BertConfig, TFBertForSequenceClassification
import requests
import json
# tokenizer = AutoTokenizer.from_pretrained("jplu/tf-xlm-r-ner-40-lang")#, use_fast = False)
# model = AutoModelForTokenClassification.from_pretrained("jplu/tf-xlm-r-ner-40-lang", from_tf=True)

class BertPretrained():
    def __init__(self, model_path = "./out_kcbert_base_221115_2332/checkpoint-4050000/"):
        from transformers import BertForMaskedLM, BertTokenizer
        import torch
        with open(model_path + 'config.json', 'r') as f:
            config = json.load(f)
        self.model = BertForMaskedLM(config)
        self.model = self.model.load_state_dict(torch.load(model_path + 'pytorch_model.bin'))
        self.tokenizer = BertTokenizer.from_pretrained(
            "beomi/kcbert-base",
            do_lower_case=False,
        )
        self.mlm = pipeline(
            "fill-mask",
            model = self.model,
            tokenizer = self.tokenizer,
            framework = "pt"
        )
    def inference(self, text):
        
        

class tfxml():
    def __init__(self):
        self.nlp_ner = pipeline(
            "ner",
            model="jplu/tf-xlm-r-ner-40-lang",
            tokenizer=(
                'jplu/tf-xlm-r-ner-40-lang',  
                {"use_fast": True}),
            framework="tf"
        )
    def inference(self, text):
        output = self.nlp_ner(text)
        temp_text = text
        n = 0
        for out in output:
                temp_text = temp_text[:out['end']+n] + "[{}]".format(out['entity']) + temp_text[out['end']+n:]
                n += len(out['entity'])+2
        return temp_text

class xlmr():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
        self.model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
        self.classifier = pipeline("ner", model=model, tokenizer=tokenizer)
    def inference(self, text):
        return self.classifier(text)

class kcbert():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
        self.config = BertConfig.from_json_file("kcbert/KcBERT-Finetune/ckpt/kcbert-base-naver-ner-ckpt/checkpoint-25000/config.json")
        self.model = AutoModelForTokenClassification.from_pretrained("kcbert/KcBERT-Finetune/ckpt/kcbert-base-naver-ner-ckpt/checkpoint-25000/pytorch_model.bin", config=self.config)
        # self.model = TFBertForSequenceClassification.from_pretrained("kcbert/KcBERT-Finetune/ckpt/kcbert-base-naver-ner-ckpt/checkpoint-25000/pytorch_model.bin", from_pt=True, config=self.config)
        # model = AutoModelWithLMHead.from_pretrained("beomi/kcbert-base")
        self.classifier = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
    def inference(self, text):
        return self.classifier(text)

class spacy_example():
    def __init__(self):
        self.nlp = spacy.load('ko_core_news_lg')
        # from spacy.pipeline.ner import DEFAULT_NER_MODEL
        # config = {
        # "moves": None,
        # "update_with_oracle_cut_size": 100,
        # "model": DEFAULT_NER_MODEL,
        # "incorrect_spans_key": "incorrect_spans",
        # }
        # self.nlp.add_pipe("ner", config=config)
    def inference(self, t):
        doc = self.nlp(t)
        # print([(w) for w in doc])
        # print([(w.ent_type_) for w in doc])
        text = [(w) for w in doc] #[서울, 전역에, 내리는, 소나기는, 현대, 기아, 모터스의, 주가, 상승에, 긍정적인, 영향을, 끼쳤다, .]
        output = [(w.ent_type_) for w in doc] #['LC', '', '', '', '', 'OG', 'OG', '', '', '', '', '', '']
        bio = [(w.ent_iob_) for w in doc]
        return text, output, bio
class letrAPI():
    def __init__(self):
        self.URL_LETR = 'https://api.letr.ai'
        self.URL_LETR_NER = self.URL_LETR + '/ner'
        self.LETR_KEY = 'LETR09SLKG4Z2BFZ8NIQAT8YFCNRYXSPGL6J'
    def ner(self, sentences, language_code = 'ko'):
        print('\nStart LETR NER ...')
        headers = {'x-api-key': self.LETR_KEY}
        # headers = {'x-api-key': LETR_KEY}
        values = {'sentences': sentences, 'language_code': language_code}
        result = requests.post(self.URL_LETR_NER, json=json.loads(json.dumps(values)), headers=headers)
        output = json.loads(result.text)
        if not 'data' in output:
            print('NOTHING AFTER NER-API....')
            print(output)
            print(result.status_code)
            return None
        return output

if __name__ == "__main__":    
    # nlp_ner = pipeline(
    #     "ner",
    #     model="jplu/tf-xlm-r-ner-40-lang",
    #     tokenizer=(
    #         'jplu/tf-xlm-r-ner-40-lang',  
    #         {"use_fast": True}),
    #     framework="tf"
    # )

    
    # from train_1028 import train as kcbert
    # text = "서울 전역에 내리는 소나기는 현대 기아 모터스의 주가 상승에 긍정적인 영향을 끼쳤다."
    # spcaye = spacy_example()
    # letr = letrAPI()
    import pandas as pd
    from tqdm import tqdm
    import time
    import random
    dir = "15rogowntpdy.json"
    with open(dir, "r", encoding = 'utf-8') as f:
        test_dataset = json.load(f)
    # ckpt_name = 'epoch=2-val_loss=0.06'
    # kcbert_model = kcbert.inference('C:/nlpbook/checkpoint-ner/{}.ckpt'.format(ckpt_name))
    # test_dataset = pd.read_csv('corpus/new_corpus_no_overlap_no_drop_test_data_4_1109.csv', sep=',')
    # kcbert_model = kcbert()
    # lines = []
    lines = test_dataset['ko_original'].values.tolist()
    # print(len(lines))
    # total_output = []
    # for i in tqdm(range(0, len(lines), 30)):
    #     idx = i+30
    #     if idx >= len(lines): idx = len(lines)
    #     while(True):
    #         output = letr.ner(lines[i:idx])
    #         time.sleep(random.randint(1,10))
    #         if output != None:
    #             break
    #     # print(type(output))
    #     # print(output['data'])
    #     total_output += output['data']
    #     # print(total_output[i:i+5 if i+5 <= len(lines) else -1])
    #     if idx == len(lines): break
    # xlmr = xlmr()
    final_outputs = []
    for line in tqdm(test_dataset['ko_original']):
        outputs = []
        # lines.append(line.strip())
        output = kcbert_model.inference(line)
        # output = kcbert_model.inference_fn(line)
        # for result in output['result']:
        #     outputs.append(result)
        final_outputs.append({'sentence' : line, 'result' : output})
        # print(outputs)
        # break
        # output = xlmr.inference(line)
        # t, o, b = spcaye.inference(line)
        # lines.append({'ko_original' : line, 'tokens' : t, 'output' : o, 'bio_tags' : b})
        # break
    # print(lines[:1])
    output = pd.DataFrame(final_outputs)
    output.to_csv('output_kc_bert_{}_1110.csv'.format('ckpt_name'), index=False)
    # print('done')
            # f.write("{}\n".format(str(output)))
            # print(output)
        # string 형식
        # n = 0
        # for out in output:
        #     text = text[:out['end']+n] + "[{}]".format(out['entity']) + text[out['end']+n:]
        #     n += len(out['entity'])+2 #+ len(out['word'])
        #     print(out['entity'], out['word'], out['start'], out['end'])
        # print(text)pip