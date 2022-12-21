import enum
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re

label_map = [
    '[CLS]',
    '[SEP]',
    '[PAD]',
    '[MASK]',
    'O',
    'B-PER',
    'B-ARF',
    'B-ORG',
    'B-DAT',
    'B-ANM',
    'B-CVL',
    'B-THR',
    'B-LOC',
    'B-QTT',
    'B-TRM',
    'B-STF',
    'B-TIM',
    'B-PLT',
    'B-EVT',
    'B-MAT',
    'I-PER',
    'I-ARF',
    'I-ORG',
    'I-DAT',
    'I-ANM',
    'I-CVL',
    'I-THR',
    'I-LOC',
    'I-QTT',
    'I-TRM',
    'I-STF',
    'I-TIM',
    'I-PLT',
    'I-EVT',
    'I-MAT'
    ]


def BIO_labelstudio(file_dir, out_dir):
    import json
    label_map = {
     'PER' : ['PERSON', 'PS'],
     'STF' : ['FD', 'STUDY_FIELD'],
     'THR' : ['TR', 'THEORY'],
     'ARF' : ['AF', 'AFA', 'WORK_OF_ART', 'AFW', 'PRODUCT', 'ARTIFACTS', 'WORK', 'ARRIFACTS'],
     'ORG' : ['OGG', 'ORG', 'ORGANIZATION'],
     'CVL' : ['CV', 'CIVILIZATION'],
     'LOC' : ['LC','LCG', 'LCP', 'LOCATION'],
     'DAT' : ['DT', 'DATE'],
     'TIM' : ['TI', 'TIME'],
     'QTT' : ['QT', 'QUANTITY'],
     'EVT' : ['EV', 'EVENT'],
     'ANM' : ['AM', 'ANIMAL'],
     'PLT' : ['PT', 'PLANT'],
     'MAT' : ['MT', 'MATERIAL'],
     'TRM' : ['TM','TMI', 'TMIG', 'TMM', 'TERM']
    }

    with open(file_dir, "r", encoding="utf-8") as file:
        corpus = json.load(file)
    new_data = []
    for data in tqdm(corpus):
        ko = data["ko"]
        ko = [k for k in ko]
        for label in data["label"]:
            if data.get("sn"):
                start = label["start"]
                end = label["end"]
                # text = label["text"]
                tag = label.get("labels")
                if tag: tag = tag[0]
                else:
                    print("cannot find label")
                    print(data["ko"])
                    print(label)
                    tag = input("new_label : ")
                    if tag == "C" or tag == "c":
                        continue
            else:
                start = label["position"][0]
                end = label["position"][1]
                # text = label["value"]
                tag = label["tag"].split("_")[0]
            flag = False
            for k, v in label_map.items():
                if tag in v:
                    tag = k
                    flag = True
            if not flag:
                print("cannot find tag : {}".format(tag))
                old_tag = tag
                tag = input("new_tag : ")
                label_map[tag].append(old_tag)
                print("label_map : {} added to {}".format(old_tag, tag))
            # print(start, end)
            for i in range(start, end):
                try:
                    if i == start: ko[i] = "B-" + tag
                    else: ko[i] = "I-" + tag
                except IndexError as e:
                    print(e)
                    print(ko)
                    print(data["ko"])
                    print(start, end)
                    print(label)
                    ko = ["O" for k in ko]
                    break
        # print(ko)
        ko = [k if len(k) > 1 and k.split("-")[1] in label_map.keys() else "O" for k in ko]
        new_data.append({"data" : data, "output" : ko})
        # print(ko)
        # print("-----")
        
        
    with open(out_dir, "w", encoding="utf-8") as file:
        json.dump(new_data, file, indent=2, ensure_ascii=False)
    print('done')
            

def BIO_corpus(file_dir, result_file_dir):
    result = Path(file_dir)
    df = pd.read_csv(result)
    new_df = {'ko_original' : [], 'output' : []}
    for i, outputs in tqdm(enumerate(df['ner.tags'])): #['ko_original', 'ner.tags'...]
        ko_original = df['ko_original'][i]
        # spaces = []
        # for i, ko in enumerate(ko_original):
        #     if ko == ' ':
        new_output = ['O' for _ in range(len(ko_original))]
        
        output = eval(outputs)
        for tag in output:
            pos = eval(str(tag['position']))
            if tag['tag'] != 'O':
                for j in range(int(pos[0]), int(pos[1])):
                    # print(tag)
                    # print(output)
                    if j == pos[0]:
                        # print(ko_original)
                        # print(tag)
                        # print(new_output, j)
                        try:
                            new_output[j] = "{}-B".format(tag['tag'])
                        except IndexError:
                            print(new_output)
                            print(df['ner.text'][i])
                            print(tag)
                            print(j)
                            jn = input()
                            new_output[j-int(jn)] = "{}-B".format(tag['tag'])
                    else:
                        # print(df['ko_original'][i])
                        try:
                            new_output[j] = "{}-I".format(tag['tag'])
                        except IndexError:
                            print(new_output)
                            print(df['ner.text'][i])
                            print(tag)
                            print(j)
                            jn = input()
                            new_output[j-int(jn)] = "{}-I".format(tag['tag'])
        new_df['ko_original'].append(ko_original)
        new_df['output'].append(new_output)
    new_df = pd.DataFrame(new_df)
    new_df.to_csv('{}'.format(result_file_dir))

def BIO_xlmr(file_dir, result_file_dir):
    result = Path(file_dir)
    df = pd.read_csv(result)
    
    for i, outputs in tqdm(enumerate(df['ner'])): #['ko_original', 'output']
        ko_original = df['sentence'][i]
        new_output = ['O' for _ in range(len(ko_original))]

        output = eval(outputs)
        # if len(output) > 0:
        for tag in output: #['entity', 'score', 'index', 'word', 'start', 'end']
            for j in range(eval(str(tag['position']))[0],eval(str(tag['position']))[1]):
                if j == eval(str(tag['position']))[0]:
                    new_output[j] = "B-{}".format(tag['entity'])
                else: new_output[j] = "I-{}".format(tag['entity'])
        df.loc[i, 'ner'] = str(new_output)
    df.to_csv('{}'.format(result_file_dir))
    print('done')
def BIO_roberta_json(file_dir, result_file_dir): #[{'token': '세', 'predicted_tag': 'B-PER', 'top_prob': '0.9992'}, {'token': '##찬', 'predicted_tag': 'I-PER', 'top_prob': '0.9994'}, {'token': '##이', 'predicted_tag': 'I-PER', 'top_prob': '0.9557'}, {'token': '##인가', 'predicted_tag': 'O', 'top_prob': '0.9939'}, {'token': '봐', 'predicted_tag': 'O', 'top_prob': '0.9999'}, {'token': '.', 'predicted_tag': 'O', 'top_prob': '0.9999'}]
    result = Path(file_dir)
    import json
    with open(result, "r", encoding="utf-8") as file:
        df = json.load(file)
    label_map = [
    '[CLS]',
    '[SEP]',
    '[PAD]',
    '[MASK]',
    'O',
    'B-PER',
    'B-ARF',
    'B-ORG',
    'B-DAT',
    'B-ANM',
    'B-CVL',
    'B-THR',
    'B-LOC',
    'B-QTT',
    'B-TRM',
    'B-STF',
    'B-TIM',
    'B-PLT',
    'B-EVT',
    'B-MAT',
    'I-PER',
    'I-ARF',
    'I-ORG',
    'I-DAT',
    'I-ANM',
    'I-CVL',
    'I-THR',
    'I-LOC',
    'I-QTT',
    'I-TRM',
    'I-STF',
    'I-TIM',
    'I-PLT',
    'I-EVT',
    'I-MAT',
    'I-WOA',
    'B-WOA',
    'I-PRD',
    'B-PRD'
    ]
    label_map = [
    '[CLS]',
    '[SEP]',
    '[PAD]',
    '[MASK]',
    'O',
    'PER-B',
    'FLD-B',
    'AFW-B',
    'ORG-B',
    'LOC-B',
    'CVL-B',
    'DAT-B',
    'TIM-B',
    'NUM-B',
    'EVT-B',
    'ANM-B',
    'PLT-B',
    'MAT-B',
    'TRM-B',
    'PER-I',
    'FLD-I',
    'AFW-I',
    'ORG-I',
    'LOC-I',
    'CVL-I',
    'DAT-I',
    'TIM-I',
    'NUM-I',
    'EVT-I',
    'ANM-I',
    'PLT-I',
    'MAT-I',
    'TRM-I'
    ]
    label_map = [
        "O",
        "B-PER",
        "B-PRD",
        "B-ORG", 
        "B-WOA", 
        "I-PER",
        "I-PRD",
        "I-ORG",
        "I-WOA"
    ]
    aaa = 0
    ndf = []
    def replace_list(new_sentence, token, predicted_tag):
        flag = False
        for token_idx, t in enumerate(token):
            for i, s in enumerate(new_sentence):
                if t == s:
                    if token_idx > 0 and predicted_tag[0] == "B":
                        new_sentence[i] = "I" + predicted_tag[1:]
                    else: new_sentence[i] = predicted_tag
                    flag = True
                    break
        if not flag: print("error ", new_sentence, token)
        return new_sentence, flag
    for data_idx, data in tqdm(enumerate(df)):
        sentence = data['sentence']
        result = data['result']
        new_sentence = [s for s in sentence]
        tags = []
        n = 0
        for i, re in enumerate(result): 
            token = re['token']
            predicted_tag = re['predicted_tag']
            token = token.replace("##", "")
            if token == "[UNK]": #, "[SEP]", "[CLS]", "[PAD]", "[MASK]"]:
                token = ">"
            new_sentence, flag = replace_list(new_sentence, token, predicted_tag)
        
        for sen_idx, s in enumerate(new_sentence):
            if s not in label_map:
                if s == " ":
                    if new_sentence[sen_idx+1][0] == "I":
                        new_sentence[sen_idx] = new_sentence[sen_idx+1]
                    else: new_sentence[sen_idx] = "O"
                else:
                    new_sentence[sen_idx] = "O"
                    # print("error 000\t", new_sentence)

        ndf.append({"sentence" : sentence, "result" : result, "output" : new_sentence})
    with open(result_file_dir, "w", encoding="utf-8") as file:
        json.dump(ndf, file, indent=2, ensure_ascii=False)
    print('done')
def BIO_bert_json(file_dir, result_file_dir): #[{'token': '세', 'predicted_tag': 'B-PER', 'top_prob': '0.9992'}, {'token': '##찬', 'predicted_tag': 'I-PER', 'top_prob': '0.9994'}, {'token': '##이', 'predicted_tag': 'I-PER', 'top_prob': '0.9557'}, {'token': '##인가', 'predicted_tag': 'O', 'top_prob': '0.9939'}, {'token': '봐', 'predicted_tag': 'O', 'top_prob': '0.9999'}, {'token': '.', 'predicted_tag': 'O', 'top_prob': '0.9999'}]
    result = Path(file_dir)
    import json
    with open(result, "r", encoding="utf-8") as file:
        df = json.load(file)
    # label_map = [
    # '[CLS]',
    # '[SEP]',
    # '[PAD]',
    # '[MASK]',
    # 'O',
    # 'B-PER',
    # 'B-ARF',
    # 'B-ORG',
    # 'B-DAT',
    # 'B-ANM',
    # 'B-CVL',
    # 'B-THR',
    # 'B-LOC',
    # 'B-QTT',
    # 'B-TRM',
    # 'B-STF',
    # 'B-TIM',
    # 'B-PLT',
    # 'B-EVT',
    # 'B-MAT',
    # 'I-PER',
    # 'I-ARF',
    # 'I-ORG',
    # 'I-DAT',
    # 'I-ANM',
    # 'I-CVL',
    # 'I-THR',
    # 'I-LOC',
    # 'I-QTT',
    # 'I-TRM',
    # 'I-STF',
    # 'I-TIM',
    # 'I-PLT',
    # 'I-EVT',
    # 'I-MAT',
    # 'I-WOA',
    # 'B-WOA',
    # 'I-PRD',
    # 'B-PRD'
    # ]
    label_map = [
    # '[CLS]',
    # '[SEP]',
    # '[PAD]',
    # '[MASK]',
    'O',
    'B-PER',
    'B-CVL',
    'B-DAT',
    'B-QTT',
    'B-THR',
    'B-ANM',
    'B-ORG',
    'B-TRM',
    'B-STF',
    'B-ARF',
    'B-LOC',
    'B-TIM',
    'B-MAT',
    'B-PLT',
    'B-EVT',
    'I-PER',
    'I-CVL',
    'I-DAT',
    'I-QTT',
    'I-THR',
    'I-ANM',
    'I-ORG',
    'I-TRM',
    'I-STF',
    'I-ARF',
    'I-LOC',
    'I-TIM',
    'I-MAT',
    'I-PLT',
    'I-EVT',
    ]
    # label_map = [
    # '[CLS]',
    # '[SEP]',
    # '[PAD]',
    # '[MASK]',
    # 'O',
    # 'PER-B',
    # 'FLD-B',
    # 'AFW-B',
    # 'ORG-B',
    # 'LOC-B',
    # 'CVL-B',
    # 'DAT-B',
    # 'TIM-B',
    # 'NUM-B',
    # 'EVT-B',
    # 'ANM-B',
    # 'PLT-B',
    # 'MAT-B',
    # 'TRM-B',
    # 'PER-I',
    # 'FLD-I',
    # 'AFW-I',
    # 'ORG-I',
    # 'LOC-I',
    # 'CVL-I',
    # 'DAT-I',
    # 'TIM-I',
    # 'NUM-I',
    # 'EVT-I',
    # 'ANM-I',
    # 'PLT-I',
    # 'MAT-I',
    # 'TRM-I'
    # ]
    # label_map = [
    #     "O",
    #     "B-PER",
    #     "B-PRD",
    #     "B-ORG", 
    #     "B-WOA", 
    #     "I-PER",
    #     "I-PRD",
    #     "I-ORG",
    #     "I-WOA"
    # ]
    aaa = 0
    ndf = []
    def replace_list(new_sentence, token, predicted_tag):
        flag = False
        for token_idx, t in enumerate(token):
            for i, s in enumerate(new_sentence):
                if t == s:
                    if token_idx > 0 and predicted_tag[0] == "B":
                        new_sentence[i] = "I" + predicted_tag[1:]
                    else: new_sentence[i] = predicted_tag
                    flag = True
                    break
        if not flag: print("error ", new_sentence, token)
        return new_sentence, flag
    for data_idx, data in tqdm(enumerate(df)):
        sentence = data['sentence']
        result = data['result']
        new_sentence = [s for s in sentence]
        tags = []
        n = 0
        for i, re in enumerate(result): 
            token = re['token']
            predicted_tag = re['predicted_tag']
            token = token.replace("##", "")
            if token == "[UNK]": #, "[SEP]", "[CLS]", "[PAD]", "[MASK]"]:
                token = ">"
            new_sentence, flag = replace_list(new_sentence, token, predicted_tag)
        
        for sen_idx, s in enumerate(new_sentence):
            if s not in label_map:
                if s == " ":
                    if new_sentence[sen_idx+1][0] == "I":
                        new_sentence[sen_idx] = new_sentence[sen_idx+1]
                    else: new_sentence[sen_idx] = "O"
                else:
                    new_sentence[sen_idx] = "O"
                    # print("error 000\t", new_sentence)

        ndf.append({"sentence" : sentence, "result" : result, "output" : new_sentence})
    with open(result_file_dir, "w", encoding="utf-8") as file:
        json.dump(ndf, file, indent=2, ensure_ascii=False)
    print('done')

def BIO_kcbert(file_dir, result_file_dir): #[{'token': '세', 'predicted_tag': 'B-PER', 'top_prob': '0.9992'}, {'token': '##찬', 'predicted_tag': 'I-PER', 'top_prob': '0.9994'}, {'token': '##이', 'predicted_tag': 'I-PER', 'top_prob': '0.9557'}, {'token': '##인가', 'predicted_tag': 'O', 'top_prob': '0.9939'}, {'token': '봐', 'predicted_tag': 'O', 'top_prob': '0.9999'}, {'token': '.', 'predicted_tag': 'O', 'top_prob': '0.9999'}]
    result = Path(file_dir)
    df = pd.read_csv(result)
    label_map = [
    '[CLS]',
    '[SEP]',
    '[PAD]',
    '[MASK]',
    'O',
    'B-PER',
    'B-ARF',
    'B-ORG',
    'B-DAT',
    'B-ANM',
    'B-CVL',
    'B-THR',
    'B-LOC',
    'B-QTT',
    'B-TRM',
    'B-STF',
    'B-TIM',
    'B-PLT',
    'B-EVT',
    'B-MAT',
    'I-PER',
    'I-ARF',
    'I-ORG',
    'I-DAT',
    'I-ANM',
    'I-CVL',
    'I-THR',
    'I-LOC',
    'I-QTT',
    'I-TRM',
    'I-STF',
    'I-TIM',
    'I-PLT',
    'I-EVT',
    'I-MAT',
    'I-WOA',
    'B-WOA',
    'I-PRD',
    'B-PRD'
    ]
    label_map = [
    '[CLS]',
    '[SEP]',
    '[PAD]',
    '[MASK]',
    'O',
    'PER-B',
    'FLD-B',
    'AFW-B',
    'ORG-B',
    'LOC-B',
    'CVL-B',
    'DAT-B',
    'TIM-B',
    'NUM-B',
    'EVT-B',
    'ANM-B',
    'PLT-B',
    'MAT-B',
    'TRM-B',
    'PER-I',
    'FLD-I',
    'AFW-I',
    'ORG-I',
    'LOC-I',
    'CVL-I',
    'DAT-I',
    'TIM-I',
    'NUM-I',
    'EVT-I',
    'ANM-I',
    'PLT-I',
    'MAT-I',
    'TRM-I'
    ]

    aaa = 0
    for result, sentence in tqdm(zip(df['result'], df['sentence'])): #['ko_original', 'output']
        new_sentence = [s for s in sentence]
        result = eval(result)
        # if len(output) > 0:
        tags = []
        for i, token in enumerate(result): #['entity', 'score', 'index', 'word', 'start', 'end']
            # print(token)
            token = eval(str(token))
            # rptoken = token['token'].replace("#", "") for finetuned kcbert
            rptoken = token['word'].replace("#", "") # for naver ner
            n = 0
            for t in range(len(rptoken)):
                # if token['token'] == '[UNK]' or token['token'] == '[SEP]': for fintuned kcbert
                if token['word'] == '[UNK]' or token['word'] == '[SEP]': # for naver ner
                    n += int(len(sentence)/len(result)+1)
                    continue
                # print(new_sentence)
                try:
                    # print(rptoken[t])
                    idx = new_sentence.index(rptoken[t] if n+3 < len(new_sentence) else len(new_sentence)-1)
                except ValueError as e:
                    idx = -1
                    # if idx < 0 and token['predicted_tag'] != 'O': for finetuend
                    if idx < 0 and token['entity'] != 'O': # for naver ner
                        for nn, ns in enumerate(new_sentence):
                            if rptoken[t] == ns:
                                idx = nn
                    if idx == -1: continue
                # new_sentence[idx] = token['predicted_tag'] if t == 0 or token['predicted_tag'] == 'O' else "I" + token['predicted_tag'][1:]
                # new_sentence[idx] = token['entity'] if t == 0 or token['entity'] == 'O' else "I" + token['entity'][1:]
                new_sentence[idx] = token['entity'][2:] + '-B' if t == 0 or token['entity'] == 'O' else token['entity'][2:] + "I"
                # tags.append("I" + token['predicted_tag'][1:] if t > 0 and token['predicted_tag'].strip() != 'O' and token['predicted_tag'].strip() != '[UNK]' else token['predicted_tag'])
        # print(new_sentence)
        # print(sentence)
        # break
        # w = 0
        # for i, s in enumerate(sentence):
        #     if s == " ":
        #         tags = tags[:i+w] + ['O'] + tags[i+w:]
        #         w += 1
        for nn, ns in enumerate(new_sentence):
            nnmin = nn-1 if nn > 0 else nn
            nnplus = nn+1 if nn+1 < len(new_sentence) else nn
            # if new_sentence[nnmin][1:] == new_sentence[nnplus][1:] and new_sentence[nnmin] != '' and new_sentence[nnplus] != '' and new_sentence[nnmin] != 'O' and new_sentence[nnplus] != 'O' :
            if new_sentence[nnmin][0] == 'B' and new_sentence[nnplus][0] == 'I' and new_sentence[nnmin][1:] == new_sentence[nnplus][1:]:
                new_sentence[nn] = 'I' + new_sentence[nnmin][1:]
            if ns == ' ' or ns not in label_map:
                new_sentence[nn] = 'O'
            
        # print(new_sentence)
        if not len(new_sentence) == len(sentence):
            print("????!!!!!!!!")
            print(new_sentence)
            print(sentence)
        #     print(sentence)
        #     print([s for s in sentence])
        #     print(tags)
        #     print(result)
        #     print(len([s for s in sentence]))
        #     print(len(tags))
            # new_output[k] = "{}-{}".format(bio_tag[k], tag) if bio_tag[k] != 'O' else "{}".format(bio_tag[k])
        df.loc[aaa, 'output'] = str(new_sentence)
        # df.loc[i, 'ko_original'] = sentence
        # if aaa == 100: break
        aaa += 1
    df.to_csv('{}'.format(result_file_dir))
    print('done')

def BIO_spacy(file_dir, result_file_dir): #"[골라봤습니다만, 요, 세, 가지의, 작품에, 대해서]","['O', 'O', 'B-QT', 'I-QT', 'O', 'O']"
    result = Path(file_dir)
    df = pd.read_csv(result)
    
    for i, outputs in tqdm(enumerate(df['output'])): #['ko_original', 'output']
        bio_tags = df['bio_tags'][i]
        ko_original = df['ko_original'][i]
        # new_output = ['O' for _ in range(len(ko_original))]

        output = eval(outputs)
        bio_tag = eval(bio_tags)
        new_output = bio_tag
        # if len(output) > 0:
        for k, tag in enumerate(output): #['entity', 'score', 'index', 'word', 'start', 'end']
            new_output[k] = "{}-{}".format(bio_tag[k], tag) if bio_tag[k] != 'O' else "{}".format(bio_tag[k])
        df.loc[i, 'output'] = new_output
    df.to_csv('{}'.format(result_file_dir))
    print('done')

def BIO_corpus_token(file_dir, result_file_dir): #"'본 제품은 한국의료기기 안전정보원으로부터 털을 제거하는 기구로 의료기기 인증을 받았습니다.'","['O', 'O', 'B-OG', 'I-OG', 'O', 'O', 'O', 'O', 'O', 'O']"
    result = Path(file_dir)
    df = pd.read_csv(result)
    new_df = {'ko_original' : [], 'output' : []}
    for i, outputs in enumerate(df['ner.tags']): #['ko_original', 'ner.tags'...]
        ko_original = df['ko_original'][i]
        wspecial = df['w/special'][i]
        n = len(ko_original) - len(wspecial)
        spaces_ko_original = []
        for i, ko in enumerate(ko_original):
            if ko == ' ': spaces_ko_original.append(i)
        new_output = ['O' for _ in ko_original.split()]
        output = eval(outputs)
        for tag in output:
            pos = eval(str(tag['position']))
            if tag['tag'] != 'O':
                start_token_idx = 0
                end_token_idx = 0
                for space in spaces_ko_original:
                    if space <= int(pos[0]-n):
                        start_token_idx += 1
                    if space <= int(pos[1]-n):
                        end_token_idx += 1
                for j in range(start_token_idx, end_token_idx+1):
                    print(wspecial,ko_original, outputs)
                    if j == start_token_idx:
                        new_output[j] = "B-{}".format(tag['tag'])                    
                    else:
                        new_output[j] = "I-{}".format(tag['tag'])
        print(new_output)
        new_df['ko_original'].append(ko_original)
        new_df['output'].append(new_output)
        break
    print(new_df)
    new_df = pd.DataFrame(new_df)
    new_df.to_csv('{}'.format(result_file_dir))

def getScores(p_dir, y_dir, is_p_json = True, is_y_json = True):
    import numpy as np
    import sklearn.metrics as metrics
    import json

    label_map = {"B-PER" : "B-QTT",
    "B-CVL" : "B-ARF",
    "B-DAT" : "B-PER",
    "B-QTT" : "B-TRM",
    "B-THR" : "B-CVL",
    "B-ANM" : "B-EVT",
    "B-ORG" : "B-ORG",
    "B-TRM" : "B-DAT",
    "B-STF" : "B-LOC",
    'B-ARF' : "B-THR",
    'B-LOC' : "B-ANM",
    'B-TIM' : "B-MAT",
    'B-MAT' : "B-TIM",
    'B-PLT' : "B-PLT",
    'B-EVT' : "B-STF",
    'I-PER' : "I-QTT",
    'I-CVL' : "I-ARF",
    'I-DAT' : "I-PER",
    'I-QTT' : "I-TRM",
    'I-THR' : "I-CVL",
    'I-ANM' : "I-EVT",
    'I-ORG' : "I-ORG",
    'I-TRM' : "I-DAT",
    'I-STF' : "I-LOC",
    'I-ARF' : "I-THR",
    'I-LOC' : "I-ANM",
    'I-TIM' : "I-MAT",
    'I-MAT' : "I-TIM",
    'I-PLT' : "I-PLT",
    'I-EVT' : "I-STF",
    "O" : "O"
    }
    labels = []
    if is_p_json:
        with open(p_dir, "r", encoding="utf-8") as file:
            p = json.load(file)
        p = [i['output'] for i in p]
        p_new = []
        for pp in p:
            for ppp in pp:
                # p_new.append(label_map[ppp] if label_map[ppp] == "O" else label_map[ppp].split("-")[1])
                p_new.append(ppp if ppp == "O" else ppp.split("-")[1])
                if ppp != "O" and ppp.split("-")[1] not in labels:
                    labels.append(ppp.split("-")[1])
                # p_new.append(ppp)
    else:
        p = pd.read_csv(p_dir)
        p = p['output'].values.tolist()
        p_new = []
        for pp in p:
            for ppp in eval(pp):
                p_new.append(ppp)
    if is_y_json:
        with open(y_dir, "r", encoding="utf-8") as file:
            y = json.load(file)
        y = [i['output'] for i in y]
        y_new = []
        for pp in y:
            for ppp in pp:
                y_new.append(ppp if ppp == "O" else ppp.split("-")[1])
                # y_new.append(ppp)
    else:
        y = pd.read_csv(y_dir)
        y = y['output'].values.tolist()
        y_new = []
        for pp in y:
            for ppp in eval(pp):
                y_new.append(ppp)
    
    
    p = p_new
    y = y_new
    print('model : {}'.format('roberta-base CSE tokenizer'))
    # print('model : {}'.format('roberta-base trained BPE tokenizer'))
    # print('model : {}'.format('bert-base_t_KcBERT_mos'))
    # print('model : {}'.format('KcBERT'))
    # print('lr=1e-5\titer4750000')
    # print('lr=1e-5\tepoch=6\tval_loss=0.18')
    # print('lr=1e-5\tepoch=6\tval_loss=0.18')
    # print('learning rate : 0.00005, epoch : 3, batch_size : 32, val_loss : 0.06')
    print('accuracy', metrics.accuracy_score(y,p))
    print('precision', metrics.precision_score(y,p,average='micro'))
    print('recall', metrics.recall_score(y,p,average='micro'))
    print('f1 micro', metrics.f1_score(y,p,average='micro'))
    print('f1 macro', metrics.f1_score(y,p,average='macro'))

    print(metrics.classification_report(y,p,labels = labels,zero_division=True))
    # print(metrics.confusion_matrix(y,p))
if __name__ == "__main__":
    # BIO_corpus('new_corpus/new_corpus_no_overlap_no_drop_test_data_4_1109.csv', 'new_corpus/new_corpus_no_overlap_no_drop_test_data_4_1109_BIO.csv')
    # BIO_corpus("corpus/new_corpus_no_overlap_no_drop_1110.csv", "new_corpus_naver_ner_1110_BIO.csv")
    # BIO_xlmr("corpus/output_test_letr_API_no_cardinal.csv", "output_test_letr_API_no_cardinal.csv_renewed.csv")
    # BIO_spacy("output_test_spcay.csv", "output_test_spcay_renewed.csv")
    # BIO_corpus_token("corpus/new_corpus_no_overlap_no_drop_spacy221027.csv", "corpus/new_corpus_no_overlap_no_drop_spacy221027_special_test_0.1_BIO.csv")

    # BIO_labelstudio("./tagging/test.json","./tagging/ntest_BIO.json")
    name = "old_Roberta_mos_16617"
    # BIO_bert_json(name + ".json", name + "_BIO.json")
    # edit_label_map(name+".json")

    # getScores("output_kc_bert_epoch=2-val_loss=0.06_1109.csv", "corpus/new_corpus_no_overlap_no_drop_train_data_4_1109.csv")
    # getScores("roberta_KoSimCSE_23024_BIO.json", "new_corpus_no_overlap_no_drop_test_data_4_1109_BIO.csv", is_y_json=False)
    # getScores("output_test_xlm-roberta-large-finetuned-conll03-english_renewed.csv", 'corpus/new_corpus_no_overlap_no_drop_xlmr_test_0.1_BIO_221026.csv')
    # getScores("roberta_base_t_trainedbpe_ep7_33049_BIO.json", "./tagging/test_BIO.json")
    # getScores(name + "_BIO.json", "./tagging/test_BIO.json")
    getScores(name + "_BIO.json", "./tagging/Dataset_H_mo_s/test_BIO.json")
    # getScores("bert_base_kcbert_23024_BIO.json", "./tagging/Dataset_H_mo_s/test_BIO.json")
    