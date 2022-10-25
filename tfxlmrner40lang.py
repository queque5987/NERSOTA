from transformers import pipeline

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

if __name__ == "__main__":    
    nlp_ner = pipeline(
        "ner",
        model="jplu/tf-xlm-r-ner-40-lang",
        tokenizer=(
            'jplu/tf-xlm-r-ner-40-lang',  
            {"use_fast": True}),
        framework="tf"
    )

    # text = "서울 전역에 내리는 소나기는 현대 기아 모터스의 주가 상승에 긍정적인 영향을 끼쳤다."
    import pandas as pd
    from tqdm import tqdm
    test_dataset = pd.read_csv('corpus/new_corpus_no_overlap_no_drop_xlmr_test_0.1.csv', sep=',')
    lines = []
    for line in tqdm(test_dataset['ko_original']):
        # lines.append(line.strip())
        output = nlp_ner(line)
        lines.append({'ko_original' : line, 'output' : output})
    print(lines[:1])
    output = pd.DataFrame(lines)
    output.to_csv('output_test_xlmr.csv', index=False)
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