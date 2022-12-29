import csv
import json
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import re
import pprint
import argparse

def to_train_bert_csv(corpus_dir : Path):
    df = pd.read_csv(corpus_dir, sep=',')
    lines = []
    for i, ko in enumerate(df['ko_original']):
        print(i/len(df['ko_original'])*100)
        try:
            ner_tags = eval(df['ner.tags'][i])
            new_line = ko
            ner_tags.sort(key=lambda x: int(eval(eval(str(x))['position'])[0]))
            temp_tag = []
            for ner_tag in ner_tags:
                ner_tag = eval(str(ner_tag))
                temp = {}
                temp['tag'] = ner_tag['tag']
                temp['value'] = ner_tag['value']
                temp['position'] = eval(ner_tag['position']) if type(ner_tag['position']) == type('') else ner_tag['position']
                temp_tag.append(temp)
            ner_tags = temp_tag

            for j, ner_tag in enumerate(ner_tags):
                tag = ner_tag['tag']
                position = ner_tag['position']
                value = ner_tag['value']
                n = len("<:{}>".format(tag))
                pos = int(position[0])
                new_line = new_line[:int(position[0])] + "<{}:{}>".format(value, tag) + new_line[int(position[1]):]
                for k in range(j, len(ner_tags)):
                    inner_ner_tag = ner_tags[k]
                    position = inner_ner_tag['position']
                    if position[0] > pos:
                        position = [post + n for post in position]
                    ner_tags[k]['position'] = position
        except TypeError as e:
            print("{} :(\n-----".format(e))
            print(ko)
            ner_tag = eval(str(ner_tag))
            print(ner_tag['position'])
            print(type(ner_tag['position']))
            print('-----\n')
            new_line = input()
        lines.append("{}␞{}".format(ko, new_line))
    with open('train_1028/ner/val.txt', 'w', encoding='utf-8') as file:
        for i, line in tqdm(enumerate(lines)):
            file.write(line + "\n" if i < len(lines)-1 else line)
    print('done')
    return 0

def to_train_bert_aihub(corpus_dir : Path, output_dir : Path):

    """
    {
      "sn": "KTOS062012215132770",
      "data_set": "일상생활및구어체",
      "domain": "일상생활",
      "subdomain": "여행",
      "ko_original": ">민호가 할 것 같아서.",
      "ko": ">민호가 할 것 같아서.",
      "mt": "I think Mino will do it.",
      "en": ">I think Minho will do it.",
      "source_language": "ko",
      "target_language": "en",
      "word_count_ko": 4,
      "word_count_en": 6,
      "word_ratio": 1.5,
      "file_name": "여행_KTOS.xlsx",
      "source": "SBS",
      "license": "open",
      "style": "구어체",
      "included_unknown_words": false,
      "ner": {
        "text": "><PERSON>민호</PERSON>가 할 것 같아서.",
        "tags": [
          {
            "tag": "PERSON",
            "value": "민호",
            "position": "[1, 3]"
          }
        ]
      }
    """
    with open(corpus_dir, "r", encoding="utf-8") as f:
        df = json.load(f)
    lines = []
    for i, data in tqdm(enumerate(df['data'])):
        try:
            ner_tags = data['ner']
            new_line = data['ko']
            temp_tag = []
            for ner_tag in ner_tags['tags']:
                temp = {}
                temp['tag'] = ner_tag['tag']
                temp['value'] = ner_tag['value']
                temp['position'] = ner_tag['position'] if type(ner_tag['position'])==type([]) else eval(str(ner_tag['position']))
                temp_tag.append(temp)
            ner_tags = temp_tag
            ner_tags.sort(key=lambda x: x['position'][0])

            for j, ner_tag in enumerate(ner_tags):
                tag = ner_tag['tag']
                position = ner_tag['position']
                value = ner_tag['value']
                n = len("<:{}>".format(tag))
                pos = int(position[0])
                new_line = new_line[:int(position[0])] + "<{}:{}>".format(value, tag) + new_line[int(position[1]):]
                for k in range(j, len(ner_tags)):
                    inner_ner_tag = ner_tags[k]
                    position = inner_ner_tag['position']
                    if position[0] > pos:
                        position = [post + n for post in position]
                    ner_tags[k]['position'] = position
        except TypeError as e:
            print("{} :(\n직접 작성해 주세요.\tex)나는 <유재석:PER>이다.\n-----".format(e))
            print(data)
            print(ner_tag)
            print('-----\n')
            new_line = input()
        lines.append("{}␞{}".format(data['ko'], new_line))
    with open(output_dir + '.txt', 'w', encoding='utf-8') as file:
        for i, line in tqdm(enumerate(lines)):
            file.write(line + "\n" if i < len(lines)-1 else line)
    print('done :\t{} sentences'.format(len(lines)))
    return 0

def to_train_bert_momal(corpus_dir : Path, output_dir : Path):

    """
    "document": [
        {
            "id": "SBRW1800000004",
            "metadata": {
                "title": "EBS 초대석 4",
                "author": "엄길청, 캐서린한",
                "publisher": "EBS",
                "date": "",
                "topic": "",
                "url": ""
            },
            "sentence": [
                {
                    "id": "SBRW1800000004.1",
                    "form": "네 여러분 안녕하십니까. 엄길청입니다.",
                    "word": [
                        {
                            "id": 1,
                            "form": "네",
                            "begin": 0,
                            "end": 1
                        },
                        {
                            "id": 2,
                            "form": "여러분",
                            "begin": 2,
                            "end": 5
                        },
                        {
                            "id": 3,
                            "form": "안녕하십니까.",
                            "begin": 6,
                            "end": 13
                        },
                        {
                            "id": 4,
                            "form": "엄길청입니다.",
                            "begin": 14,
                            "end": 21
                        }
                    ],
                    "NE": [
                        {
                            "id": 1,
                            "form": "엄길청",
                            "label": "PS_NAME",
                            "begin": 14,
                            "end": 17,
                            "kid": "00000000033281",
                            "wikiid": "1807669",
                            "URL": "https://ko.wikipedia.org/wiki/%EC%97%84%EA%B8%B8%EC%B2%AD"
                        }
                    ]
                },
                ...
    """
    upper_tag_dict = {
    'PS' : 'PER',
    'FD' : 'STF',
    'TR' : 'THR',
    'AF' : 'ARF', 'AFA' : 'ARF', 'AFW' : 'ARF',
    'OGG': 'ORG', 'ORG' : 'ORG',
    'CV' : 'CVL',
    'LC' : 'LOC', 'LCG' : 'LOC', 'LCP' : 'LOC',
    'DT' : 'DAT',
    'TI' : 'TIM',
    'QT' : 'QTT',
    'EV' : 'EVT',
    'AM' : 'ANM',
    'PT' : 'PLT',
    'MT' : 'MAT',
    'TM' : 'TRM', 'TMI' : 'TRM', 'TMIG' : 'TRM', 'TMM' : 'TRM',
    'O'  : 'O'
    }
    with open(corpus_dir, "r", encoding="utf-8") as f:
        df = json.load(f)
    lines = []
    for i, document in tqdm(enumerate(df['document'])):
        for doc in document:
            try:
                ner_tags = doc['NE']
                new_line = doc['form']
                temp_tag = []
                for ner_tag in ner_tags:
                    temp = {}
                    temp['tag'] = upper_tag_dict[ner_tag['label'].split('_')[0]]
                    temp['value'] = ner_tag['form']
                    temp['position'] = [ner_tag['begin'], ner_tag['end']]
                    temp_tag.append(temp)
                ner_tags = temp_tag
                ner_tags.sort(key=lambda x: x['position'][0])

                for j, ner_tag in enumerate(ner_tags):
                    tag = ner_tag['tag']
                    position = ner_tag['position']
                    value = ner_tag['value']
                    n = len("<:{}>".format(tag))
                    pos = int(position[0])
                    new_line = new_line[:int(position[0])] + "<{}:{}>".format(value, tag) + new_line[int(position[1]):]
                    for k in range(j, len(ner_tags)):
                        inner_ner_tag = ner_tags[k]
                        position = inner_ner_tag['position']
                        if position[0] > pos:
                            position = [post + n for post in position]
                        ner_tags[k]['position'] = position
            except TypeError as e:
                print("{} :(\n직접 작성해 주세요.\tex)나는 <유재석:PER>이다.\n-----".format(e))
                print(doc)
                print(ner_tag)
                print('-----\n')
                new_line = input()
            lines.append("{}␞{}".format(doc['form'], new_line))
    with open(output_dir + '.txt', 'w', encoding='utf-8') as file:
        for i, line in tqdm(enumerate(lines)):
            file.write(line + "\n" if i < len(lines)-1 else line)
    print('done :\t{} sentences'.format(len(lines)))
    return 0

def to_train_bert_labelstudio_concat(dirs, output_dir):
    hmap = {}
    no_tag_hmap = {}
    def update_hmap(ko, hmap):
        if hmap.get(hash(ko)):
            return False, hmap
        hmap[hash(ko)] = 1
        return True, hmap
    def to_train_bert_labelstudio(corpus_dir, no_tag_hmap, hmap):
        with open(corpus_dir, "r", encoding="utf-8") as file:
            corpus = json.load(file)
        with open("tagging/no_tag_hmap.json", "r", encoding="utf-8") as file:
            no_tag_hmap = json.load(file)
        new_corpus = []
        split = "␞"
        tag_dict = {'PER' : ['PERSON', 'PS'],
        'STF' : ['FD', 'STUDY_FIELD'],
        'THR' : ['TR', 'THEORY', 'THERORY'],
        'ARF' : ['AF', 'AFA', 'WORK_OF_ART', 'AFW', 'PRODUCT', 'ARTIFACTS', 'ARRIFACTS'],
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
        def tag_change(tag):
            n_tag = None
            for key, value in tag_dict.items():
                if tag in value:
                    n_tag = key
            if n_tag: return n_tag
            else:
                print(" unexpected tag : " + tag)
                for k in tag_dict.keys():
                    print(k, end=", ")
                while(True):
                    new_tag = input("select tag : ")
                    if tag_dict.get(new_tag):
                        tag_dict[new_tag].append(tag)
                        break
                    print("wrong tag")
            return tag_change(tag)
        count = 0
        for data in tqdm(corpus):
            if data.get("label"):
                labels = sorted(data["label"], key=lambda x:x["start"])
                ko = data["ko"]
                n = 0
                for i, label in enumerate(labels):
                    start = label["start"] + n
                    end = label["end"] + n
                    text = label.get("text")
                    if not text:
                        print("no_text_error\n--c to continue\n--or write new text")
                        print(ko)
                        print(label)
                        text = input("\tnew_text : ")
                        if text == "c" or text == "C":
                            continue
                        labels[i]["text"] = text
                    tag = label.get("labels")
                    if tag: tag = tag[0]
                    if not tag:
                        if no_tag_hmap.get(hash(text)):
                            tag = no_tag_hmap[hash(text)]
                        else:
                            print(" {}".format(ko))
                            print("{}".format(label))
                            print("no_tag_error\n--c to continue\n--or write new tag")
                            tag = input("\tnew_tag : ")
                            if tag == "c" or tag == "C":
                                continue
                            no_tag_hmap[hash(text)] = tag
                            with open("tagging/no_tag_hmap.json", "w", encoding="utf-8") as file:
                                json.dump(no_tag_hmap, file)
                        labels[i]["labels"] = [tag]
                    tag = tag_change(tag)
                    ko = ko[:start] + "<{}:{}>".format(text, tag) + ko[end:] #<희철이:PER>
                    n += len("<:{}>".format(tag))
                if data["ko"].split() == ko.split():
                    continue
                uflag, hmap = update_hmap(data["ko"], hmap)
                if uflag:
                    count += 1
                    new_corpus.append({"train_bert" : data["ko"]+split+ko, "data" : data})
            else: continue
        print("sentence : {} from {}".format(count, len(corpus)))
        return new_corpus, no_tag_hmap, hmap
    out_corpus = []
    log = 0
    for dir in dirs:
        nc, no_tag_hmap, hmap = to_train_bert_labelstudio(dir, no_tag_hmap, hmap)
        out_corpus += nc
        log += 1
        print("done {} out of {}".format(log, len(dirs)))
    print(len(out_corpus))
    print("saving whole data to {}. . .".format(output_dir))
    with open(output_dir, "w", encoding="utf-8") as file:
        json.dump([out["data"] for out in out_corpus], file, indent=2, ensure_ascii=False)
        
    import random
    random.shuffle(out_corpus)
    print("saving {} sentences to val.txt., val.json".format(len(out_corpus)//10*1))
    with open(os.path.join(output_dir, "val.txt"), "w", encoding="utf-8") as file:
        txt_lines = [out["train_bert"] for out in out_corpus[:len(out_corpus)//10*1]]
        for i, line in tqdm(enumerate(txt_lines)):
            file.write(line + "\n" if i < len(txt_lines)-1 else line)
    with open(os.path.join(output_dir, "val.json"), "w", encoding="utf-8") as file:
        json.dump([out["data"] for out in out_corpus[:len(out_corpus)//10*1]], file, indent=2, ensure_ascii=False)

    print("saving {} sentences to test.txt., test.json".format(len(out_corpus)//10*2 - len(out_corpus)//10*1))
    with open(os.path.join(output_dir, "test.txt"), "w", encoding="utf-8") as file:
        txt_lines = [out["train_bert"] for out in out_corpus[len(out_corpus)//10*2:len(out_corpus)//10*1]]
        for i, line in tqdm(enumerate(txt_lines)):
            file.write(line + "\n" if i < len(txt_lines)-1 else line)
    with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as file:
        json.dump([out["data"] for out in out_corpus[len(out_corpus)//10*1:len(out_corpus)//10*2]], file, indent=2, ensure_ascii=False)

    print("saving {} sentences to train.txt., train.json".format(len(out_corpus) - len(out_corpus)//10*2))
    with open(os.path.join(output_dir, "train.txt"), "w", encoding="utf-8") as file:
        txt_lines = [out["train_bert"] for out in out_corpus[len(out_corpus)//10*2:]]
        for i, line in tqdm(enumerate(txt_lines)):
            file.write(line + "\n" if i < len(txt_lines)-1 else line)
    with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as file:
        json.dump([out["data"] for out in out_corpus[len(out_corpus)//10*2:]], file, indent=2, ensure_ascii=False)

    print("done")

def to_upload_labelstudio(dir):
    import json
    with open(dir, "r", encoding = 'utf-8') as f:
        predicted = json.load(f)
    """
    result_out = {
        "from_name": "label",
        "to_name": "text",
        "type": "labels",
        "value": {}
    }
    value_out = {
            "start" : 0,
            "end" : 0,
            "text" : "",
            "labels" : []
        }
    out = {
        'data' : {
            "ko" : "",
            "sn" : "",
        },
        "predictions" : [
            {
                "model_version": "KcBert-finetuned",
                "result" : []
            }
        ]
    }
    """
    label_dict = {
    "PER" : "PERSON",
    "LOC" : "LOCATION",
    "ORG" : "ORGANIZATION",
    "ARF" : "ARTIFACTS",
    "DAT" : "DATE",
    "ANM" : "ANIMAL",
    "CVL" : "CIVILIZATION",
    "THR" : "THEORY",
    "QTT" : "QUANTITY",
    "TRM" : "TERM",
    "STF" : "STUDY_FIELD",
    "TIM" : "TIME",
    "PLT" : "PLANT",
    "EVT" : "EVENT",
    "MAT" : "MATERIAL"
    }
    to_save_output = []
    def pop(sentence : str, to_pop):
        for t in to_pop:
            sf = sentence.find(t)
            if sf >= 0:
                sentence = sentence[:sf] + sentence[sf+1:]
        return sentence
    
    for p in tqdm(predicted):
        sentence = p['sentence']
        
        output = []
        
        n = 0
        temp_value_format = None
        for result in p['result']:
            token = result["token"]
            tag = result["predicted_tag"]
            if tag != "O" and tag != "[SEP]":
                tag = tag[:2] + label_dict[tag[2:]]
            if tag[0] == "B":
                value_format = {
                    "start" : 0,
                    "end" : 0,
                    "text" : "",
                    "labels" : []
                }
                value_format["start"] = n
                value_format["end"] = n+len(token.replace("##", ""))
                value_format["text"] = token.replace("##", "")
                value_format["labels"] = [tag[2:]]
                temp_value_format = value_format
            if tag[0] == "I" :
                if temp_value_format and temp_value_format["labels"][0] == tag[2:]:
                    temp_value_format["end"] += len(token.replace("##", "")) if token[:2] == "##" else len(token)+1
                    temp_value_format["text"] += token.replace("##", "") if token[:2] == "##" else " " + token
                else:
                    value_format = value_format = {
                        "start" : 0,
                        "end" : 0,
                        "text" : "",
                        "labels" : []
                    }
                    value_format["start"] = n
                    value_format["end"] = n+len(token.replace("##", ""))
                    value_format["text"] = token.replace("##", "")
                    value_format["labels"] = [tag[2:]]
                    temp_value_format = value_format
            if tag == "O" and temp_value_format:
                output.append(temp_value_format)
                temp_value_format = None
            n += len(token.replace("##", ""))
            sentence = pop(sentence, token.replace("##", ""))
            if sentence and sentence[0] == " ":
                sentence = sentence[1:]
                n += 1
        if temp_value_format:
            output.append(temp_value_format)
        data_output = {
            'data' : {
                "ko" : p['sentence'],
                "sn" : p['sn'],
            },
            "predictions" : [
                {
                    "model_version": "KcBert-finetuned",
                    "result" : [{
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "value": out
                    } for out in output]
                }
            ]
        }
        to_save_output.append(data_output)
    from pprint import pprint
    print("sample -----\n")
    pprint(to_save_output[:5])
    print("\n----- sample")
    with open("{}_labelstudio_.json".format(dir.replace('.json','')), "w", encoding='utf-8') as outfile:
        json.dump(to_save_output, outfile, indent=2, ensure_ascii=False)

def pretrain_dataset(json_dir, output_dir):
    with open(json_dir, "r", encoding="utf-8") as file:
        j = json.load(file)
    out_sentences = []
    hmap = {}
    def add(sentence, out_sentences, hmap):
        if not hmap.get(hash(sentence)):
            hmap[hash(sentence)] = 1
            out_sentences.append(sentence)
        return out_sentences, hmap
    for data in j:
        if data.get("document"):
            for document in data['document']:
                if document.get("utterance"):
                    for utt in document['utterance']:
                        out_sentences, hmap = add(utt['form'], out_sentences=out_sentences, hmap=hmap)
                elif document.get("sentence"):
                    for sen in document['sentence']:
                        out_sentences, hmap = add(sen['form'], out_sentences=out_sentences, hmap=hmap)
        elif data.get("data"):
            for d in data:
                out_sentences, hmap = add(d['ko_original'], out_sentences=out_sentences, hmap=hmap)
    with open(os.path.join(output_dir, "pretrain_dataset.json"), "w", encoding='utf-8') as outfile:
        json.dump(out_sentences, outfile, indent=2, ensure_ascii=False)
    print("done : \t{}".format(len(out_sentences)))

def get_json_list(corpus_dir : Path):
    json_dir = []
    json_dir = dir_serach(corpus_dir, json_dir)
    return json_dir

def dir_serach(dir : Path, json_dir : list):
    if dir.is_file() and sub_dir.name[-3:] not in ['zip', 'pdf']:
        return Path(sub_dir)
    for sub_dir in os.scandir(dir):
        if sub_dir.is_file() and sub_dir.name[-3:] not in ['zip', 'pdf']:
            json_dir.append(Path(sub_dir))
        if sub_dir.is_dir():
            json_dir += dir_serach(sub_dir, [])
    return json_dir

def concat_datasets(dirs, output_dir):
    out_corpus = []
    hmap = {}
    is_finetuning = False
    if dirs[0].name[-3:] == "txt":
        is_finetuning = True
    for dir in dirs:
        with open(dir, "r", encoding = "utf-8") as file:
            lines = file.readlines() if is_finetuning else json.load(file)
            for line in lines:
                if not hmap[hash(line.split("␞")[0])]:
                    hmap[hash(line.split("␞")[0])] = 1
                    out_corpus.append(line)

    import random
    random.shuffle(out_corpus)

    print("saving {} sentences to val.txt or val.json".format(len(out_corpus)//10*1))
    if is_finetuning:
        with open(os.path.join(output_dir, "val.txt"), "w", encoding="utf-8") as file:
            txt_lines = [out for out in out_corpus[:len(out_corpus)//10*1]]
            for i, line in tqdm(enumerate(txt_lines)):
                file.write(line + "\n" if i < len(txt_lines)-1 else line)
    else:
        with open(os.path.join(output_dir, "val.json"), "w", encoding="utf-8") as file:
            json.dump([out for out in out_corpus[:len(out_corpus)//10*1]], file, indent=2, ensure_ascii=False)

    print("saving {} sentences to test.txt or test.json".format(len(out_corpus)//10*2 - len(out_corpus)//10*1))
    if is_finetuning:
        with open(os.path.join(output_dir, "test.txt"), "w", encoding="utf-8") as file:
            txt_lines = [out for out in out_corpus[len(out_corpus)//10*2:len(out_corpus)//10*1]]
            for i, line in tqdm(enumerate(txt_lines)):
                file.write(line + "\n" if i < len(txt_lines)-1 else line)
    else:
        with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as file:
            json.dump([out for out in out_corpus[len(out_corpus)//10*1:len(out_corpus)//10*2]], file, indent=2, ensure_ascii=False)

    print("saving {} sentences to train.txt or train.json".format(len(out_corpus) - len(out_corpus)//10*2))
    if is_finetuning:
        with open(os.path.join(output_dir, "train.txt"), "w", encoding="utf-8") as file:
            txt_lines = [out for out in out_corpus[len(out_corpus)//10*2:]]
            for i, line in tqdm(enumerate(txt_lines)):
                file.write(line + "\n" if i < len(txt_lines)-1 else line)
    else:
        with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as file:
            json.dump([out for out in out_corpus[len(out_corpus)//10*2:]], file, indent=2, ensure_ascii=False)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Korean colloquial NER Dataset ETL code by queque5987')

    parser.add_argument('-d', '--dataset', required=True, default="./dataset", help='Directory of dataset')
    parser.add_argument('-m', '--mode', required=True, default="NER", help='\'P\'for Pretrain\n\'NER\'for NER finetuning\n\'labelstudio\'for labelstudio upload\n\'concat\' for concat and distribute datasets')
    parser.add_argument('-f', '--format', required=False, default="", help='Format of dataset \'AIHub\', \'modu\', \'Labelstudio\'')
    
    args = parser.parse_args()
    if args.concat.lower() == "true": args.concat = True
    if args.concat.lower() == "false": args.concat = False
    if args.mode.lower() == "ner" and not args.format:
        print("dataset's format is required : \n")
        args.format = input()

    dirs = get_json_list(Path(args.dataset))
    for dir in dirs:
        if args.mode.lower() == "ner":
            if args.format.lower() == "aihub":
                to_train_bert_aihub(dir, dir)
            elif args.format.lower() == "modu":
                to_train_bert_momal(dir, dir)
            elif args.format.lower() == "labelstudio":
                break
        elif args.mode.lower() == "p":
            break
        elif args.mode.lower() == "labelstudio":
            to_upload_labelstudio(dir)
        elif args.mode.lower() == "concat":
            break
    if args.mode.lower() == "ner" and args.format.lower() == "labelstudio":
        to_train_bert_labelstudio_concat(dirs, os.path.join(Path(args.dataset) + "l_f_dataset.json"))
    if args.mode.lower() == "p":
        pretrain_dataset(dirs)
    if args.mode.lower() == "concat":
        concat_datasets(dirs, Path(args.dataset))
