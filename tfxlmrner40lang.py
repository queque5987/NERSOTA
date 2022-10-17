from transformers import pipeline

# from transformers import AutoTokenizer, AutoModelForTokenClassification
# tokenizer = AutoTokenizer.from_pretrained("jplu/tf-xlm-r-ner-40-lang")#, use_fast = False)
# model = AutoModelForTokenClassification.from_pretrained("jplu/tf-xlm-r-ner-40-lang", from_tf=True)

nlp_ner = pipeline(
    "ner",
    model="jplu/tf-xlm-r-ner-40-lang",
    tokenizer=(
        'jplu/tf-xlm-r-ner-40-lang',  
        {"use_fast": True}),
    framework="tf"
)

text = "서울 전역에 내리는 소나기는 현대 기아 모터스의 주가 상승에 긍정적인 영향을 끼쳤다."

output = nlp_ner(text)
n = 0
for out in output:
    text = text[:out['end']+n] + "[{}]".format(out['entity']) + text[out['end']+n:]
    n += len(out['entity'])+2 #+ len(out['word'])
    print(out['entity'], out['word'], out['start'], out['end'])
print(text)

# pip install --no-cache-dir transformers sentencepiece
# pip install tensorflow <- ?
# pip install torch 1.12.1
# https://aka.ms/vs/16/release/vc_redist.x64.exe