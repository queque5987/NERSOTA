from importlib.resources import path
import json
from operator import concat
import re
from tqdm import tqdm
from pathlib import Path


def corpus2dataset(file_dir : Path, dict_edit = None, file_list = None, do_ner = True, concat_data = False):

    if file_list == None and not concat_data:

        """
        data = {
            domain[dict_domain] = ...
            subdomain[dict_subdomain] = ...
            ner[dict_ner] = ...
            ko_original[dict_script] = ...
        }
        """
        if dict_edit == None:   dict_edit = ['domain', 'subdomain', 'ner', 'ko_original']
        else:
            dict_scripts = dict_edit[0]
            dict_domain = dict_edit[1]
            dict_subdomain = dict_edit[2]
            dict_ner = dict_edit[3]
            dict_script = dict_edit[4]

        with open(file_dir, "r", encoding = 'utf-8') as file:
            corpus = json.load(file)
            domains = {}
            info = {'sentences' : 0, 'sum_length' : 0, 'average_length' : 0}
            scripts = []
            ner = []
            if dict_scripts == '': corpus_data = corpus
            else: corpus_data = corpus[dict_scripts]
            for data in tqdm(corpus_data):
                try:
                    switch = True
                    if domains[data[dict_domain]]: switch = False
                    domains[data[dict_domain]][data[dict_subdomain]] += 1
                except:
                    if switch: domains[data[dict_domain]] = {}
                    domains[data[dict_domain]][data[dict_subdomain]] = 1
                if do_ner and data[dict_ner] != None:
                    ner.append([len(scripts) ,data[dict_ner]])
                script = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s\.\,\?]", "", data[dict_script])
                scripts.append([script, [data[dict_domain], data[dict_subdomain]]])
                info['sentences'] += 1
                info['sum_length'] += len(script)
            info['average_length'] = info['sum_length'] / info['sentences']
            info['domains'] = domains
            if do_ner and ner: dicti = {'metadata' : info, 'scripts' : scripts, 'ner' : ner}
            else : dicti = {'metadata' : info, 'scripts' : scripts}
            with open("{}_reproduced.json".format(file_dir.name.replace('.json', "")), "w", encoding='utf-8') as file:
                file.write(json.dumps(dicti, ensure_ascii=False))
            print('done')
    
    elif concat_data == True:
        import os
        sentences = ['']
        info = {'sentences' : 0, 'sum_length' : 0, 'average_length' : 0}
        for datum in tqdm(os.scandir(file_dir)):
            domain = {}
            if datum.name[-4:] == 'json':
                with open(Path(datum), "r", encoding='utf-8') as file:
                    corpus = json.load(file)
                    i = 0
                    for utter in corpus['document'][0]['utterance']:
                        cate = [a for a in corpus['metadata']['category'].split(' > ')[1:]]
                        u = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s\.\,\?]", "", utter['original_form'])
                        if len(utter['original_form']) == 0 or utter['original_form'][-1] in ['요', '다', '.']:
                            sentences[i] += u
                            sentences[i] = [sentences[i], cate]
                            i += 1
                            sentences.append('')
                            info['sum_length'] += len(u)
                        else:
                            sentences[i] += (u.strip() + " ")
                            info['sum_length'] += (len(u) + 1)
                        
                sentences.pop()
                info['sentences'] += len(sentences)
                try:
                    switch = True
                    if domain[cate[0]]:
                        switch = False
                        domain[cate[0]][cate[1]] += len(sentences)
                except:
                    if switch: domain[cate[0]] = {}
                    domain[cate[0]][cate[1]] = len(sentences)
        info['average_length'] = info['sum_length'] / info['sentences']
        # print(sentences[:10])
        # print(domain)
        # print(info)
        dicti = {'metadata' : info, 'scripts' : sentences}
        with open("{}_reproduced.json".format(file_dir.name.replace('.json', "")), "w", encoding='utf-8') as file:
            file.write(json.dumps(dicti, ensure_ascii=False))
        print('done!!')
    
    else:
        import os
        for datum in file_list:
            sub_file_dir = Path.joinpath(file_dir, datum)
            for i in os.scandir(sub_file_dir):
                for j in os.scandir(Path(i)):
                    if j.name[-4:] == 'json':
                        corpus2dataset(Path(j), dict_edit=['', '중분류', '소분류', '', '원문'], do_ner=False)
        print('done!')
            
            

if __name__ == "__main__":
    # corpus2dataset(Path("corpus/025.일상생활 및 구어체 한-영 번역 병렬 말뭉치 데이터/01.데이터/1.Training/라벨링데이터/TL2/일상생활및구어체_한영_train_set.json"))
    # corpus2dataset(Path("corpus/027.일상생활 및 구어체 한-중, 한-일 번역 병렬 말뭉치 데이터/01.데이터/1_Training/라벨링데이터/TL1"), file_list = ["한일", "한중"])
    corpus2dataset(Path("corpus\국립국어원 구어 말뭉치(버전 1.2)"), concat_data = True)