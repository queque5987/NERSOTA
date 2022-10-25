import csv
import json
import os
from pathlib import Path
from tqdm import tqdm

"""
우리 장훈 씨 같은 남편을 어디서 만나요.,>우리 <PERSON>장훈</PERSON> 씨 같은 남편을 어디서 만나요.,"[{'tag': 'PERSON', 'value': '장훈', 'position': '[4, 6]'}]"
"""

def write_csv(corpus_dir : Path, idx, writer):
    

    for i, j in enumerate(json_dir):
        if i == idx:
            NEs = []
            ko_original = ''
            nertext = ''
            nertags = []
            # {'tag': 'PERSON', 'value': '허경환', 'position': '[0, 3]'}
            with open(j, encoding='utf-8') as j_file:
                j_dict = json.load(j_file)

                for j_dict_doc in tqdm(j_dict['document']):
                    for j_dict_doc_sen in j_dict_doc['sentence']:
                        ko_original = j_dict_doc_sen['form']
                        # ko_original = j_dict['document'][0]['sentence'][0]['form']
                        nertext = ko_original
                        NEs = j_dict_doc_sen['NE']
                        # NEs = j_dict['document'][0]['sentence'][0]['NE']
                        n = 0
                        for ne in NEs:
                            nertext = nertext[:ne['begin']+n] + "<{}>".format(ne['label']) +  nertext[ne['begin']+n:ne['end']+n] + "</{}>".format(ne['label']) + nertext[ne['end']+n:]
                            n += (len(ne['label'])*2 + len("<></>"))
                            nertags.append({'tag' : ne['label'], 'value' : ne['form'], 'position' : [ne['begin'], ne['end']]})
                        if len(nertags) > 0 : writer.writerow({'ko_original' : ko_original, 'ner.text' : nertext, 'ner.tags' : nertags})
            
    print('done')
            
                    
            
        
    return 0

def get_json_list(corpus_dir : Path):
    json_dir = []
    for sub_dir in os.scandir(corpus_dir):
        for susub_dir in os.scandir(Path(sub_dir)):
            if susub_dir.name[0] == 'S':
                if susub_dir.name[-4:] == 'json':
                    json_dir.append(Path(susub_dir))
                else:
                    json_dir = dir_serach(susub_dir, json_dir)
    return json_dir

def dir_serach(dir : Path, json_dir : list):
    for sub_dir in os.scandir(dir):
        json_dir.append(Path(sub_dir))
    return json_dir

if __name__ == "__main__":
    corpus_dir = "corpus/NIKL_EL_2021_v1.1/국립국어원 개체명 분석 말뭉치 개체 연결 2021(버전 1.1)"
    corpus_dir = Path(corpus_dir)
    json_dir = get_json_list(corpus_dir)
    # with open("AIHub_new_ner_corpus.csv",'r', newline='', encoding='utf-8') as file:
    #     reader = csv.DictReader(file)
    #     fieldnames = []
    #     for row in reader:
    #         fieldnames = list(row.keys())
    #         break
        
        # with open("AIHub_new_ner_corpus.csv",'a', newline='', encoding='utf-8') as file:
        #     writer = csv.DictWriter(file, fieldnames=fieldnames)
        #     for i in tqdm(range(272)):
        #         write_csv(corpus_dir, i, writer)
    with open("AIHub_new_ner_corpus_221022.csv",'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['ko_original', 'ner.text', 'ner.tags'])
        for i in tqdm(range(272)):
            write_csv(corpus_dir, i, writer)