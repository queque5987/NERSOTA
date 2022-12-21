from transformers import pipeline
from tqdm import tqdm
from train_1028 import train1209 as bert_base_kcbert
from train_1028 import train1210 as roberta
from train_1028 import train1210_2 as roberta_t
# import spacy
# pip install --no-cache-dir transformers sentencepiece
# pip install tensorflow <- ?
# pip install torch 1.12.1
# https://aka.ms/vs/16/release/vc_redist.x64.exe

from transformers import AutoTokenizer, AutoModelForTokenClassification, BertConfig, TFBertForSequenceClassification, BatchEncoding
import requests
import json
import torch
# tokenizer = AutoTokenizer.from_pretrained("jplu/tf-xlm-r-ner-40-lang")#, use_fast = False)
# model = AutoModelForTokenClassification.from_pretrained("jplu/tf-xlm-r-ner-40-lang", from_tf=True)

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

class robertaMLM():
    def __init__(self, trained = False):
        from transformers import RobertaTokenizer, RobertaConfig, RobertaForMaskedLM, AutoTokenizer
        from tokenizers import ByteLevelBPETokenizer
        import torch
        ckpt_dir = "./out_roberta_base_KoSimCSEtokenizer_20221202/checkpoint-550000/"
        if not trained:
            self.tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta')
        else:
            self.tokenizer = ByteLevelBPETokenizer(
            'bpe/vocab.json',
            'bpe/merges.txt'
            )
            # self.tokenizer.mask_token = "<mask>"
            # self.tokenizer.mask_token_id = 4
            # self.tokenizer.save("workspace/tokenizer")
            # self.tokenizer = PreTrainedTokenizerFast(tokenizer_file="workspace/tokenizer/")
        with open(ckpt_dir + "config.json", "r") as file:
            self.config = json.load(file)
        self.model = RobertaForMaskedLM.from_pretrained(ckpt_dir)
        self.model.load_state_dict(torch.load(ckpt_dir + "pytorch_model.bin"), strict=False)
        # self.model.to(torch.device('cuda'))
        self.model.eval()
        self.pipeline = pipeline(
            "fill-mask",
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt"
        )
        self.masktoken = self.tokenizer.mask_token

    # def encode(self, text):
    #     e = self.tokenizer.encode(text)
    #     return BatchEncoding({"input_ids" : torch.tensor(e.ids), "attention_mask" : torch.tensor(e.attention_mask)})
    def inference(self, text):
        # e_input = self.encode(text)
        # print(e_input.input_ids.size())
        # return self.model(e_input)
        return self.pipeline(text.replace("<mask>", self.masktoken))

class KcBERTMLM():
    def __init__(self, ckpt_dir = "./checkpoint-4750000/"):
        from transformers import BertTokenizer, BertForMaskedLM
        import torch, json
        self.tokenizer = BertTokenizer.from_pretrained(
            "beomi/kcbert-base",
            do_lower_case=False,
        )
        with open(ckpt_dir + "config.json", "r") as file:
            self.config = json.load(file)
        self.model = BertForMaskedLM.from_pretrained("./checkpoint-4750000")
        # self.model.load_state_dict(torch.load(ckpt_dir + "pytorch_model.bin"))
        self.model.eval()
        self.pipeline = pipeline(
            "fill-mask",
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt"
        )
        self.masktoken = self.tokenizer.mask_token
    def inference(self, text):
        return self.pipeline(text.replace("<mask>", self.masktoken))

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
    # import pandas as pd
    # from tqdm import tqdm
    # import time
    # import random
    # # ckpt_name = 'epoch=2-val_loss=0.06'
    # # kcbert_model = kcbert.inference('C:/nlpbook/checkpoint-ner/{}.ckpt'.format(ckpt_name))
    # import pandas as pd
    # test_dataset = pd.read_csv("./dataset/new_corpus_no_overlap_no_drop_test_data_4_1109.csv", sep=',')
    # kcbert_model = kcbert()
    # kcbert_model = ftd_kcbert.inference(ckpt_dir="epoch=2-val_loss=0.14.ckpt", label_map_dir="label_map.txt")
    j = "./ner_mo_s/test.json"
    model = roberta.roberta()
    # model = roberta_t.roberta()
    # model = bert_base_kcbert.kcbert_inference()
    # model = bert_base_kcbert.inference()
    with open(j,'r', encoding='utf-8') as j_file:
        lines = json.load(j_file)
    # print(lines[0])
    lines = [line["ko"] for line in lines]
    # lines = []
    # lines = test_dataset['ko_original'].values.tolist()
    print(lines[0])
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
    # roberta = KcBERTMLM()
    # print(roberta.inference("서울의 수도는 <mask>입니다."))
    final_outputs = []
    count = 0
    print("total {} lines".format(len(lines)))
    for i, line in tqdm(enumerate(lines)):
    # #     outputs = []
    # #     # lines.append(line.strip())
    # #     output = kcbert_model.inference_fn(line)
    # #     # output = kcbert_model.inference_fn(line)
        # output = roberta.inference(line)
        output = model.inference_fn(line)
        final_outputs.append(output)
        count += 1
        # if count == 10: break
        # break
    # #     # for result in output['result']:
    # #     #     outputs.append(result)
    # #     final_outputs.append({'sentence' : line, 'ner' : output})
    # #     # print(outputs)
    # #     # break
    # #     # output = xlmr.inference(line)
    # #     # t, o, b = spcaye.inference(line)
    # #     # lines.append({'ko_original' : line, 'tokens' : t, 'output' : o, 'bio_tags' : b})
    # #     # break
    # # # print(lines[:1])
    #     if i%50000 == 0:
    #         print(final_outputs[0])
    #         with open("./mlm_output/bert_base_normal{}.json".format(count), "w", encoding='utf-8') as outfile:
    #             json.dump(final_outputs, outfile, indent=2, ensure_ascii=False)
    #         count += 1
    #         final_outputs = []
    import random
    print(final_outputs[random.randint(0, len(final_outputs)-1)])
    with open("./dataset/old_Roberta_mos_{}.json".format(count), "w", encoding='utf-8') as outfile:
        json.dump(final_outputs, outfile, indent=2, ensure_ascii=False)
    print('done')
    # final_outputs = []
    # output = pd.DataFrame(final_outputs)
    # output.to_csv('output_kc_bert_{}_1110.csv'.format('ckpt_name'), index=False)
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