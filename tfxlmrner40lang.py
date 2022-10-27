from transformers import pipeline
import spacy
# pip install --no-cache-dir transformers sentencepiece
# pip install tensorflow <- ?
# pip install torch 1.12.1
# https://aka.ms/vs/16/release/vc_redist.x64.exe

from transformers import AutoTokenizer, AutoModelForTokenClassification
tokenizer = AutoTokenizer.from_pretrained("jplu/tf-xlm-r-ner-40-lang")#, use_fast = False)
model = AutoModelForTokenClassification.from_pretrained("jplu/tf-xlm-r-ner-40-lang", from_tf=True)

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

if __name__ == "__main__":    
    # nlp_ner = pipeline(
    #     "ner",
    #     model="jplu/tf-xlm-r-ner-40-lang",
    #     tokenizer=(
    #         'jplu/tf-xlm-r-ner-40-lang',  
    #         {"use_fast": True}),
    #     framework="tf"
    # )

    

    # text = "서울 전역에 내리는 소나기는 현대 기아 모터스의 주가 상승에 긍정적인 영향을 끼쳤다."
    spcaye = spacy_example()
    import pandas as pd
    from tqdm import tqdm
    test_dataset = pd.read_csv('corpus/new_corpus_no_overlap_no_drop_spacy221027.csv', sep=',')
    lines = []
    xlmr = xlmr()
    for line in tqdm(test_dataset['ko_original']):
        # lines.append(line.strip())
        # output = nlp_ner(line)
        # output = xlmr.inference(line)
        t, o, b = spcaye.inference(line)
        lines.append({'ko_original' : line, 'tokens' : t, 'output' : o, 'bio_tags' : b})
        # break
    print(lines[:1])
    output = pd.DataFrame(lines)
    output.to_csv('output_test_xlm-roberta-large-finetuned-conll03-english.csv', index=False)
    print('done')
            # f.write("{}\n".format(str(output)))
            # print(output)
        # string 형식
        # n = 0
        # for out in output:
        #     text = text[:out['end']+n] + "[{}]".format(out['entity']) + text[out['end']+n:]
        #     n += len(out['entity'])+2 #+ len(out['word'])
        #     print(out['entity'], out['word'], out['start'], out['end'])
        # print(text)pip