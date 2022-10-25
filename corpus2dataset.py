import json
import re
from tqdm import tqdm
from pathlib import Path
import csv
import sys

<<<<<<< HEAD
=======

>>>>>>> a1117d7c41fa6491d8832dc5546821d912dc3bb1
def corpus2dataset(file_dir : Path, dict_edit = None, file_list = None, do_ner = True, concat_data = False, save_point = 0):

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
<<<<<<< HEAD
        with open("{}_reproduced.csv".format(file_dir.name.replace(".json", "")), "w", encoding = 'utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['scripts', 'domains'])
            sentences = ['']
            info = {'sentences' : 0, 'sum_length' : 0, 'average_length' : 0, 'domains' : {}}
            domain = {}
            
            lll = 0
            for datum in tqdm(os.scandir(file_dir)):
                if lll != 0 and lll < (save_point-1)*900:
                    lll += 1
                    continue
                if lll == save_point*900-1: break
=======
        sentences = ['']
        info = {'sentences' : 0, 'sum_length' : 0, 'average_length' : 0, 'domains' : {}}
        domain = {}
        # save_point = 1
        lll = 0
        with open("{}_reproduced.csv".format(file_dir.name.replace('.json', "")), "w", encoding='utf-8') as file:
            import csv
            writer = csv.DictWriter(file, fieldnames=['scripts', 'domains'])
            for datum in tqdm(os.scandir(file_dir)):
                # if save_point != 0 and lll <= 990*(save_point-1):
                #     lll += 1
                #     continue
                # if lll == 990*save_point:
                #     break
>>>>>>> a1117d7c41fa6491d8832dc5546821d912dc3bb1
                if datum.name[-4:] == 'json':
                    with open(Path(datum), "r", encoding='utf-8') as file:
                        corpus = json.load(file)
                        i = 0
                        for utter in corpus['document'][0]['utterance']:
                            cate = [a for a in corpus['metadata']['category'].split(' > ')[1:]]
                            u = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s\.\,\?]", "", utter['original_form'])
                            if len(utter['original_form']) == 0 or utter['original_form'][-1] in ['요', '다', '.', '?'] or utter == corpus['document'][0]['utterance'][-1]:
                                sentences[i] += u
                                writer.writerow({'scripts' : sentences[i], 'domains' : cate})
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
<<<<<<< HEAD
                        if len(cate) < 2:
                            print(cate)
                            cate.append(cate[0])
                        domain[cate[0]][cate[1]] = len(sentences)
                print(lll)
                lll += 1
            # info['domains'] = domain
            # info['average_length'] = info['sum_length'] / info['sentences']
            # dicti = {'metadata' : info, 'scripts' : sentences[:len(sentences)//4]}
            # dicti2 = {'metadata' : info, 'scripts' : sentences[len(sentences)//4:len(sentences)//4*2]}
            # dicti3 = {'metadata' : info, 'scripts' : sentences[len(sentences)*2//4*2:len(sentences)//4*3]}
            # dicti4 = {'metadata' : info, 'scripts' : sentences[len(sentences)*2//4*3:]}
            # print(len(dicti['scripts']))

        
            # for sen in sentences:
            #     writer.writerow({'scripts' : sen[0], 'domains' : sen[1]})
        # with open("{}_reproduced_info1.json".format(file_dir.name.replace('.json', "")), "w", encoding='utf-8') as file:
        #     file.write(json.dumps(info, ensure_ascii=False))
        # with open("{}_1_reproduced.json".format(file_dir.name.replace('.json', "")), "w", encoding='utf-8') as file:
        #     file.write(json.dumps(dicti, ensure_ascii=False))
        # with open("{}_2_reproduced.json".format(file_dir.name.replace('.json', "")), "w", encoding='utf-8') as file:
        #     file.write(json.dumps(dicti2, ensure_ascii=False))
        # with open("{}_3_reproduced.json".format(file_dir.name.replace('.json', "")), "w", encoding='utf-8') as file:
        #     file.write(json.dumps(dicti3, ensure_ascii=False))
        # with open("{}_4_reproduced.json".format(file_dir.name.replace('.json', "")), "w", encoding='utf-8') as file:
            # file.write(json.dumps(dicti4, ensure_ascii=False))
=======
                        domain[cate[0]][cate[1]] = len(sentences)
                lll += 1
            # info['domains'] = domain
            # info['average_length'] = info['sum_length'] / info['sentences']
            # print(sentences[:10])
            # print(domain)
            # print(info)
            dicti = {'metadata' : info, 'scripts' : sentences}
            print(lll)
        
            # i = 0
            # for sen in sentences:
            #     print("{}/{}".format(i, len(sentences)))
            #     writer.writerow({'scripts' : sen[0], 'domains' : sen[1]})
            #     i += 1
        # with open("{}_reproduced_info.json".format(file_dir.name.replace('.json', "")), "w", encoding='utf-8') as file:
        #     file.write(json.dumps(info, ensure_ascii=False))
>>>>>>> a1117d7c41fa6491d8832dc5546821d912dc3bb1
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

def dataset_xlm():
    import os

    dat_categories = ["c_event", "fm_drama", "fs_drama", "enter", "culture", "history"]
    cate = ['시사', '가족드라마', '현대드라마', '예능', '교양지식', '역사극']
    domain = {'시사' : 0, '가족드라마' : 0, '현대드라마' : 0, '예능' : 0, '교양지식' : 0, '역사극' : 0}
    cur_cate = 0    
    for dat_category in dat_categories:
        texts = []
        info = {'sentences' : 0, 'sum_length' : 0, 'average_length' : 0, 'domains' : {}}
        dat_categories = ["c_event", "fm_drama", "fs_drama", "enter", "culture", "history"]
        cate = ['시사', '가족드라마', '현대드라마', '예능', '교양지식', '역사극']
        domain = {'시사' : 0, '가족드라마' : 0, '현대드라마' : 0, '예능' : 0, '교양지식' : 0, '역사극' : 0}

        print("processing --- {}".format(dat_category))
        data_dir = Path("corpus/방송대본요약/1.Training/원천데이터/TS1/{}".format(dat_category))
        data = os.scandir(data_dir)
        for dat in tqdm(data):
            for subcate in os.scandir(dat):
                with open(subcate, "r", encoding = 'utf-8') as file:
                    d_json = json.load(file)
                    dddd = d_json['Meta']['passage'].strip().split("\n")
                    for ddd in dddd:
                        index = ddd.find(']')
                        if (dat_category == "fm_drama" or dat_category == "fs_drama") and ddd[:index] == "해설":
                            continue
                        g_index = ddd.find('(')
                        ge_index = ddd.find(')')
                        if g_index > -1 and ge_index > -1:
                            ddd = ddd[:g_index] + ddd[ge_index+1:]
                        # while(True):
                        #     g_index = dd.find('(')
                        #     ge_index = dd.find(')')
                        #     if g_index > -1 and ge_index > -1:
                        #         dd = dd[:g_index] + dd[ge_index+1:]
                        #     else: break
                        texts.append([ddd[index+1:], ['방송', cate[cur_cate]]])
                        domain[cate[cur_cate]] += 1
                        info['sentences'] += 1
                        info['sum_length'] += len(ddd[index+1:])
        cur_cate += 1
        to_del = []
        for key, dom in domain.items():
            if dom == 0:
                to_del.append(key)
        for key in to_del:
            del domain[key]
        domain = {'방송' : domain}
        info['domains'] = domain
        info['average_length'] = info['sum_length'] / info['sentences']
        dicti = {'metadata' : info, 'scripts' : texts}
        with open("방송대본요약데이터_{}_reproduced.json".format(dat_category), "w", encoding='utf-8') as file:
            file.write(json.dumps(dicti, ensure_ascii=False))

    print("done")

if __name__ == "__main__":
    # corpus2dataset(Path("corpus/025.일상생활 및 구어체 한-영 번역 병렬 말뭉치 데이터/01.데이터/1.Training/라벨링데이터/TL2/일상생활및구어체_한영_train_set.json"))
    # corpus2dataset(Path("corpus/027.일상생활 및 구어체 한-중, 한-일 번역 병렬 말뭉치 데이터/01.데이터/1_Training/라벨링데이터/TL1"), file_list = ["한일", "한중"])
<<<<<<< HEAD
    for i in range(1, 300):
        corpus2dataset(Path("corpus\국립국어원 구어 말뭉치(버전 1.2)"), concat_data = True, save_point = i)
=======
    # corpus2dataset(Path("corpus\국립국어원 구어 말뭉치(버전 1.2)"), concat_data = True)
    # for i in range(40):
    corpus2dataset(Path("corpus\국립국어원 일상 대화 말뭉치 2020(버전 1.2)"), concat_data = True, save_point=0)
>>>>>>> a1117d7c41fa6491d8832dc5546821d912dc3bb1
    # dataset_xlm()