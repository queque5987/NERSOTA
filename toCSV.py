import csv
import json
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import re

"""
우리 장훈 씨 같은 남편을 어디서 만나요.,>우리 <PERSON>장훈</PERSON> 씨 같은 남편을 어디서 만나요.,"[{'tag': 'PERSON', 'value': '장훈', 'position': '[4, 6]'}]"
"""

def write_csv(json_dir):
    ko_originals = []
    nertexts = []
    nertagses = []
    hmap = {}
    aj_sum = 0
    aj_count = 0
    for j in tqdm(json_dir):
        # {'tag': 'PERSON', 'value': '허경환', 'position': '[0, 3]'}
        with open(j,'r', encoding='utf-8') as j_file:
            j_dict = json.load(j_file)
            
            for j_dict_doc in j_dict['document']:
                pub = j_dict_doc['metadata']['publisher']
                try:
                    if hmap[hash(pub)][0] == pub:
                        hmap[hash(pub)][1] += len(j_dict_doc['sentence'])
                except:
                    hmap[hash(pub)] = [pub, len(j_dict_doc['sentence'])]
                
                for j_dict_doc_sen in j_dict_doc['sentence']:
                    
                    NEs = []
                    ko_original = ''
                    nertext = ''
                    nertags = []
                    ko_original = j_dict_doc_sen['form']
                    nertext = ko_original
                    NEs = j_dict_doc_sen['NE']
                    n = 0
                    for ne in NEs:
                        nertext = nertext[:ne['begin']+n] + "<{}>".format(ne['label']) +  nertext[ne['begin']+n:ne['end']+n] + "</{}>".format(ne['label']) + nertext[ne['end']+n:]
                        n += (len(ne['label'])*2 + len("<></>"))
                        nertags.append({'tag' : ne['label'], 'value' : ne['form'], 'position' : [ne['begin'], ne['end']]})
                    if len(nertags) > 0 :
                        ko_originals.append(ko_original)
                        nertagses.append(nertags)
                        nertexts.append(nertext)
                    aj_sum += len(re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9]", "", ko_original))
                    aj_count += 1
    print(aj_sum/aj_count)
    df = pd.read_csv("AIHub_new_ner_corpus - 복사본.csv", sep = ',')
    ndf = pd.DataFrame(ko_originals, columns=['ko_original'])
    ndf['ner.text'] = nertexts
    ndf['ner.tags'] = nertagses
    udf = df.append(ndf)
    # udf.to_csv("new_corpus/{}3.csv".format("new_corpus"),index=False)
    print(hmap.values())
    return 0
def to_train_bert(corpus_dir : Path, mode : str):
    df = pd.read_csv(corpus_dir, sep=',')
    lines = []
    for i, ko in enumerate(df['ko_original']):
        print(i/len(df['ko_original'])*100)
        try:
            ner_tags = eval(df['ner.tags'][i])
            # n = 0
            new_line = ko
            # print(ner_tags)
            # ner_tags.sort(key=lambda x: int(eval(eval(str(x))['position'])[0]))
            # print(ner_tags)
            temp_tag = []
            for ner_tag in ner_tags:
                ner_tag = eval(str(ner_tag))
                temp = {}
                temp['tag'] = ner_tag['tag']
                temp['value'] = ner_tag['value']
                # print()
                temp['position'] = eval(ner_tag['position']) if type(ner_tag['position']) == type('') else ner_tag['position']
                temp_tag.append(temp)
            ner_tags = temp_tag

            for j, ner_tag in enumerate(ner_tags):
                tag = ner_tag['tag']
                position = ner_tag['position']
                value = ner_tag['value']
                # print(ner_tag)
                # print(new_line)
                n = len("<:{}>".format(tag))
                pos = int(position[0])
                new_line = new_line[:int(position[0])] + "<{}:{}>".format(value, tag) + new_line[int(position[1]):]
                for k in range(j, len(ner_tags)):
                    # print(ner_tags)
                    # print(k)
                    inner_ner_tag = ner_tags[k]
                    position = inner_ner_tag['position']
                    if position[0] > pos:
                        position = [post + n for post in position]
                    ner_tags[k]['position'] = position
                # print(new_line)
        except TypeError as e:
            print("{} :(\n-----".format(e))
            print(ko)
            ner_tag = eval(str(ner_tag))
            print(ner_tag['position'])
            print(type(ner_tag['position']))
            print('-----\n')
            new_line = input()
        lines.append("{}␞{}".format(ko, new_line))
        # if i == 50: break
    with open('{}.txt'.format(mode), 'w', encoding='utf-8') as file:
        for i, line in tqdm(enumerate(lines)):
            file.write(line + "\n" if i < len(lines)-1 else line)
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

def find_overlap(dir : Path):

    df = pd.read_csv(dir, sep = ',')
    to_del = []
    hmap = {}
    for i, d in enumerate(df.ko_original):
        try:
            if hmap[hash(d)].strip() == d.strip():
                # print("{}\n{}\n----------".format(hmap[hash(d)], d))
                to_del.append(i)
        except:
            hmap[hash(d)] = d
    print(len(df)) #297556
    df_del = df.drop(to_del)
    print(len(df_del)) #230244
    df_del.to_csv("new_corpus/{}.csv".format("new_corpus_no_overlap"),index=False)

def find_overlap_token(dir : Path, do_concat = False, do_drop = False, name = "", concat_tag_dict = {
     'PER' : ['PERSON', 'PS'],
     'STF' : ['FD', 'STUDY_FIELD'],
     'THR' : ['TR', 'THEORY'],
     'ARF' : ['AF', 'AFA', 'WORK_OF_ART', 'AFW', 'PRODUCT', 'ARTIFACTS'],
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
    }, drop_tag_dict = {
    'PERSON' : ['PERSON', 'PS'],
    'ORG' : ['OGG', 'ORG'],
    'PRODUCT' : ['AFW', 'PRODUCT'],
    'WORK_OF_ART' : ['AFA', 'WORK_OF_ART']
    }, no_drop_do_O = False):

    if do_concat == do_drop: return 0
    tag_list = """
    PS_NAME, PS_CHARACTER, PS_PET, FD_SCIENCE, FD_SOCIAL_SCIENCE, FD_MEDICINE, FD_ART, FD_HUMANITIES, FD_OTHERS, TR_SCIENCE, TR_SOCIAL_SCIENCE, 
    TR_MEDICINE, TR_ART, TR_HUMANITIES, TR_OTHERS, AF_BUILDING, AF_CULTURAL_ASSET, AF_ROAD, 
    AF_TRANSPORT, AF_MUSICAL_INSTRUMENT, 
    AF_WEAPON, AFA_DOCUMENT, AFA_PERFORMANCE, 
    AFA_VIDEO, AFA_ART_CRAFT, AFA_MUSIC, 
    AFW_SERVICE_PRODUCTS, AFW_OTHER_PRODUCTS, OGG_ECONOMY, OGG_EDUCATION, OGG_MILITARY, 
    OGG_MEDIA, OGG_SPORTS, OGG_ART, OGG_MEDICINE, 
    OGG_RELIGION, OGG_SCIENCE, OGG_LIBRARY, OGG_LAW, 
    OGG_POLITICS, OGG_FOOD, OGG_HOTEL, OGG_OTHERS, LCP_COUNTRY, LCP_PROVINCE, LCP_COUNTY, 
    LCP_CITY, LCP_CAPITALCITY, LCG_RIVER, LCG_OCEAN, 
    LCG_BAY, LCG_MOUNTAIN, LCG_ISLAND, 
    LCG_CONTINENT, LC_SPACE, LC_OTHERS, CV_CULTURE, CV_TRIBE, CV_LANGUAGE, CV_POLICY, 
    CV_LAW, CV_CURRENCY, CV_TAX, CV_FUNDS, CV_ART, 
    CV_SPORTS, CV_SPORTS_POSITION, CV_SPORTS_INST, 
    CV_PRIZE, CV_RELATION, CV_OCCUPATION, 
    CV_POSITION, CV_FOOD, CV_DRINK, CV_FOOD_STYLE,
    CV_CLOTHING, CV_BUILDING_TYPE, DT_DURATION, DT_DAY, DT_WEEK, DT_MONTH, 
    DT_YEAR, DT_SEASON, DT_GEOAGE, DT_DYNASTY, 
    DT_OTHERS, TI_DURATION, TI_HOUR, TI_MINUTE, TI_SECOND, 
    TI_OTHERS, QT_AGE, QT_SIZE, QT_LENGTH, QT_COUNT, 
    QT_MAN_COUNT, QT_WEIGHT, QT_PERCENTAGE, 
    QT_SPEED, QT_TEMPERATURE, QT_VOLUME, QT_ORDER,
    QT_PRICE, QT_PHONE, QT_SPORTS, QT_CHANNEL, 
    QT_ALBUM, QT_ADDRESS, QT_OTHERS, EV_ACTIVITY, EV_WAR_REVOLUTION, EV_SPORTS, 
    EV_FESTIVAL, EV_OTHERS, AM_INSECT, AM_BIRD, AM_FISH, AM_MAMMALIA, 
    AM_AMPHIBIA, AM_REPTILIA, AM_TYPE, AM_PART, 
    AM_OTHERS, PT_FRUIT, PT_FLOWER, PT_TREE, PT_GRASS, PT_TYPE, 
    PT_PART, PT_OTHERS, MT_ELEMENT, MT_METAL, MT_ROCK, MT_CHEMICAL, TM_COLOR, TM_DIRECTION, TM_CLIMATE, TM_SHAPE,
    TM_CELL_TISSUE_ORGAN, TMM_DISEASE, TMM_DRUG, 
    TMI_HW, TMI_SW, TMI_SITE, TMI_EMAIL, TMI_MODEL, 
    TMI_SERVICE, TMI_PROJECT, TMIG_GENRE, TM_SPORTS
    """.split(',')
    tag_list = [i.strip() for i in tag_list]
    
    df = pd.read_csv(dir, sep = ',')
    to_del = []
    hmap = {}
    for di, d in tqdm(enumerate(df['ner.tags'])):
        # print("{} / {} - {}%".format(di, len(df['ner.tags']), di/len(df['ner.tags'])*100))
        tags = (eval(d))
        texts = df['ner.text'][di]
        to_del_tag = []
        for ti, tag in enumerate(tags):
            if do_concat:
                t = tag['tag'].split('_')[0] #if tag['tag'] in tag_list else tag['tag']
                for key, value in concat_tag_dict.items():
                    if t in value:
                        texts = texts.replace(tag['tag'], key)
                        tags[ti]['tag'] = key
                        break
                
                t = tag['tag']
                try:
                    if hmap[hash(t)][0] == t:
                        hmap[hash(t)][1] += 1
                        continue
                except:
                    hmap[hash(t)] = [t, 1]
            
            if do_drop:
                # t = tag['tag']
                t = tag['tag'].split('_')[0] if tag['tag'] in tag_list else tag['tag']
                exists = False
                if t == 'AFA':
                    print('afa is', end = " ")
                for key, value in drop_tag_dict.items():
                    if t in value:
                        texts = texts.replace(tag['tag'], key)
                        tags[ti]['tag'] = key
                        exists = True
                        break
                if t == 'AFA':
                    print('true' if exists else 'false')
                    print(tags)
                if not exists:
                    to_replace = ""
                    # if no_drop_do_O: tags[ti]['tag'] = 'O'
                    texts = texts.replace("<{}>".format(tag['tag']), to_replace)
                    texts = texts.replace("</{}>".format(tag['tag']), to_replace)
                    to_del_tag.append(tag)
                    # if tag['tag'] == 'CV_RELATION':
                    #     print(to_del_tag)
                    # print(tag)
                else:
                    t = tag['tag']
                    try:
                        if hmap[hash(t)].strip() == t.strip():
                            continue
                    except:
                        hmap[hash(t)] = t
        if do_drop:
            # if len(tags) == len(to_del_tag) and not no_drop_do_O:
            #     to_del.append(di)
            #     # continue
            # else:
                # print(tags)
                for tdt in to_del_tag:
                    # if tdt['tag'] == 'CV_RELATION':
                    #     print('why')
                    tags.remove(tdt)
                # print(tags)

        df.loc[di, 'ner.text'] = texts
        df.loc[di, 'ner.tags'] = str(tags)
    print(hmap.values())
    if do_concat:
        df.to_csv("corpus/{}_{}.csv".format("new_corpus_no_overlap_concat", name),index=False)
    if do_drop:
        df.to_csv("corpus/{}_{}.csv".format("new_corpus_no_overlap_no_drop", name),index=False)
        # df_del = df.drop(to_del)
        # df_del.to_csv("new_corpus/{}_{}.csv".format("new_corpus_no_overlap_drop", name),index=False)
    """
    PERSON(인명), STUDY_FIELD(학문 분야 및 학파), THEORY(이론), ARTIFACTS(인공물),
    ORGANIZATION(기관 및 조직), LOCATION(지명), CIVILIZATION(문명), DATE(날짜),
    TIME(시간), QUANTITY(수량), EVENT(사건 및 사고), ANIMAL(동물), PLANT(식물),
    MATERIAL(물질), TERM(기타 용어)

    CONCAT -----
    [PERSON - 'PERSON' == 'PS',
     STUDY FIELD - 'FD', 
     THEORY - 'TR',
     ARTIFACTS - 'AF', ('AFA' == 'WORK_OF_ART'), ('AFW' == 'PRODUCT'),
     ORGANIZATION - 'OGG' == 'ORG',
     CIVILIZATION - 'CV',
     LOCATION - 'LC','LCG', 'LCP',
     DATE - 'DT',
     TIME - 'TI',
     QUANTITY - 'QT',
     EVENT - 'EV',
     ANIMAL - 'AM',
     PLANT - 'PT',
     MATERIAL - 'MT'
     TERM - 'TM','TMI', 'TMIG', 'TMM']

    DROP -----
    [PERSON - 'PERSON', 'PS_',
     ORG == 'OGG_', 'ORG',
     PRODUCT == 'AFW_', 'PRODUCT',
     WORK_OF_ART == 'AFA_', 'WORK_OF_ART']

    spacy??
    [PS - 'PERSON' == 'PS',
        PS person
     OG - 'OGG' == 'ORG',
        OG organization
     LC - 'LC','LCG', 'LCP',
        LC location
     DT - 'DT', 
        DT date
     TI - 'TI',
        TI time
     QT - 'QT',
        QT quantity]
    
    --LETR--
    'PERSON' : ['PERSON', 'PS_NAME', 'PS_CHARACTER', 'PS_PET'],
    'NORP' : ['OGG_RELIGION', 'OGG_POLITICS'],
    'FAC' : ['AF_BUILDING', 'AF_CULTURAL_ASSET', 'AF_ROAD',
        'AF_TRANSPORT',],
    'ORG' : ['OGG_ECONOMY', 'OGG_EDUCATION', 'OGG_MILITARY',
        'OGG_MEDIA', 'OGG_SPORTS', 'OGG_ART', 'OGG_MEDICINE',
        'OGG_SCIENCE', 'OGG_LIBRARY', 'OGG_LAW',
        'OGG_FOOD', 'OGG_HOTEL', 'OGG_OTHERS', 'ORG'],
    'GPE' : ['LCP_COUNTRY', 'LCP_PROVINCE', 'LCP_COUNTY',
        'LCP_CITY', 'LCP_CAPITALCITY'],
    'LOC' : ['LCG_RIVER', 'LCG_OCEAN',
        'LCG_BAY', 'LCG_MOUNTAIN', 'LCG_ISLAND',
        'LCG_CONTINENT', 'LC_SPACE', 'LC_OTHERS'],
    'PRODUCT' :  ['AF_MUSICAL_INSTRUMENT', 'AF_WEAPON',
        'AFW_SERVICE_PRODUCTS', 'AFW_OTHER_PRODUCTS', 'PRODUCT',
        'PT_FRUIT', 'PT_FLOWER', 'PT_TREE', 'PT_GRASS', 'PT_TYPE',
        'PT_PART', 'PT_OTHERS'],
    'EVENT' : ['EV_ACTIVITY', 'EV_WAR_REVOLUTION', 'EV_SPORTS',
        'EV_FESTIVAL', 'EV_OTHERS'],
    'WORK_OF_ART' : ['AFA_DOCUMENT', 'AFA_PERFORMANCE',
        'AFA_VIDEO', 'AFA_ART_CRAFT', 'AFA_MUSIC', 'WORK_OF_ART'],
    'LAW' : ['CV_POLICY', 'CV_LAW', 'CV_CURRENCY', 'CV_TAX', 'CV_FUNDS'], 
    'LANGUAGE' :  ['CV_LANGUAGE'],
    'DATE' : ['DT_DURATION', 'DT_DAY', 'DT_WEEK', 'DT_MONTH',
        'DT_YEAR', 'DT_SEASON', 'DT_GEOAGE', 'DT_DYNASTY',
        'DT_OTHERS'],
    'TIME' : ['TI_DURATION', 'TI_HOUR', 'TI_MINUTE', 'TI_SECOND',
        'TI_OTHERS'],
    'PERCENT' : ['QT_PERCENTAGE'],
    'MONEY' : ['QT_PRICE'],
    'QUANTITY' : ['QT_AGE', 'QT_SIZE', 'QT_LENGTH', 'QT_COUNT',
        'QT_MAN_COUNT', 'QT_WEIGHT',
        'QT_SPEED', 'QT_TEMPERATURE', 'QT_VOLUME', 'QT_ORDER',
        'QT_PHONE', 'QT_SPORTS', 'QT_CHANNEL',
        'QT_ALBUM', 'QT_ADDRESS', 'QT_OTHERS'],
    'ORDINAL' : ['QT_ORDER']

    [PERSON - 'PERSON' == 'PS',
     STUDY FIELD - 'FD', 
     THEORY - 'TR',
     ARTIFACTS - 'AF', ('AFA' == 'WORK_OF_ART'), ('AFW' == 'PRODUCT'),
     ORGANIZATION - 'OGG' == 'ORG',
     CIVILIZATION - 'CV',
     LOCATION - 'LC','LCG', 'LCP',
     DATE - 'DT',
     TIME - 'TI',
     QUANTITY - 'QT',
     EVENT - 'EV',
     ANIMAL - 'AM',
     PLANT - 'PT',
     MATERIAL - 'MT'
     TERM - 'TM','TMI', 'TMIG', 'TMM']

    """
def find_overlap_token_letr(dir : Path, concat_tag_dict, drop_tag_dict, do_concat = False, do_drop = False, name = "", no_drop_do_O = False):

    if do_concat == do_drop: return 0
    tag_list = """
    PS_NAME, PS_CHARACTER, PS_PET, FD_SCIENCE, FD_SOCIAL_SCIENCE, FD_MEDICINE, FD_ART, FD_HUMANITIES, FD_OTHERS, TR_SCIENCE, TR_SOCIAL_SCIENCE, 
    TR_MEDICINE, TR_ART, TR_HUMANITIES, TR_OTHERS, AF_BUILDING, AF_CULTURAL_ASSET, AF_ROAD, 
    AF_TRANSPORT, AF_MUSICAL_INSTRUMENT, 
    AF_WEAPON, AFA_DOCUMENT, AFA_PERFORMANCE, 
    AFA_VIDEO, AFA_ART_CRAFT, AFA_MUSIC, 
    AFW_SERVICE_PRODUCTS, AFW_OTHER_PRODUCTS, OGG_ECONOMY, OGG_EDUCATION, OGG_MILITARY, 
    OGG_MEDIA, OGG_SPORTS, OGG_ART, OGG_MEDICINE, 
    OGG_RELIGION, OGG_SCIENCE, OGG_LIBRARY, OGG_LAW, 
    OGG_POLITICS, OGG_FOOD, OGG_HOTEL, OGG_OTHERS, LCP_COUNTRY, LCP_PROVINCE, LCP_COUNTY, 
    LCP_CITY, LCP_CAPITALCITY, LCG_RIVER, LCG_OCEAN, 
    LCG_BAY, LCG_MOUNTAIN, LCG_ISLAND, 
    LCG_CONTINENT, LC_SPACE, LC_OTHERS, CV_CULTURE, CV_TRIBE, CV_LANGUAGE, CV_POLICY, 
    CV_LAW, CV_CURRENCY, CV_TAX, CV_FUNDS, CV_ART, 
    CV_SPORTS, CV_SPORTS_POSITION, CV_SPORTS_INST, 
    CV_PRIZE, CV_RELATION, CV_OCCUPATION, 
    CV_POSITION, CV_FOOD, CV_DRINK, CV_FOOD_STYLE,
    CV_CLOTHING, CV_BUILDING_TYPE, DT_DURATION, DT_DAY, DT_WEEK, DT_MONTH, 
    DT_YEAR, DT_SEASON, DT_GEOAGE, DT_DYNASTY, 
    DT_OTHERS, TI_DURATION, TI_HOUR, TI_MINUTE, TI_SECOND, 
    TI_OTHERS, QT_AGE, QT_SIZE, QT_LENGTH, QT_COUNT, 
    QT_MAN_COUNT, QT_WEIGHT, QT_PERCENTAGE, 
    QT_SPEED, QT_TEMPERATURE, QT_VOLUME, QT_ORDER,
    QT_PRICE, QT_PHONE, QT_SPORTS, QT_CHANNEL, 
    QT_ALBUM, QT_ADDRESS, QT_OTHERS, EV_ACTIVITY, EV_WAR_REVOLUTION, EV_SPORTS, 
    EV_FESTIVAL, EV_OTHERS, AM_INSECT, AM_BIRD, AM_FISH, AM_MAMMALIA, 
    AM_AMPHIBIA, AM_REPTILIA, AM_TYPE, AM_PART, 
    AM_OTHERS, PT_FRUIT, PT_FLOWER, PT_TREE, PT_GRASS, PT_TYPE, 
    PT_PART, PT_OTHERS, MT_ELEMENT, MT_METAL, MT_ROCK, MT_CHEMICAL, TM_COLOR, TM_DIRECTION, TM_CLIMATE, TM_SHAPE,
    TM_CELL_TISSUE_ORGAN, TMM_DISEASE, TMM_DRUG, 
    TMI_HW, TMI_SW, TMI_SITE, TMI_EMAIL, TMI_MODEL, 
    TMI_SERVICE, TMI_PROJECT, TMIG_GENRE, TM_SPORTS
    """.split(',')
    tag_list = [i.strip() for i in tag_list]
    
    df = pd.read_csv(dir, sep = ',')
    to_del = []
    hmap = {}
    for di, d in tqdm(enumerate(df['ner'])):
        # print("{} / {} - {}%".format(di, len(df['ner.tags']), di/len(df['ner.tags'])*100))
        tags = (eval(d))
        # texts = df['ner.text'][di]
        to_del_tag = []
        for ti, tag in enumerate(tags):
            if do_concat:
                # t = tag['tag'].split('_')[0] if tag['tag'] in tag_list else tag['tag']
                t = tag['entity']
                for key, value in concat_tag_dict.items():
                    if t in value:
                        # texts = texts.replace(tag['tag'], key)
                        tags[ti]['entity'] = key
                        break
                
                t = tag['entity']
                try:
                    if hmap[hash(t)][0] == t:
                        hmap[hash(t)][1] += 1
                        continue
                except:
                    hmap[hash(t)] = [t, 1]
            
            if do_drop:
                t = tag['entity']
                # t = tag['tag'].split('_')[0] if tag['tag'] in tag_list else tag['tag']
                exists = False
                for key, value in drop_tag_dict.items():
                    if t in value:
                        # texts = texts.replace(tag['tag'], key)
                        tags[ti]['entity'] = key
                        exists = True
                        break
                if not exists:
                    # to_replace = ""
                    # if no_drop_do_O: tags[ti]['tag'] = 'O'
                    # texts = texts.replace("<{}>".format(tag['tag']), to_replace)
                    # texts = texts.replace("</{}>".format(tag['tag']), to_replace)
                    to_del_tag.append(tag)
                else:
                    t = tag['entity']
                    try:
                        if hmap[hash(t)].strip() == t.strip():
                            continue
                    except:
                        hmap[hash(t)] = t
        if do_drop:
            if len(tags) == len(to_del_tag) and not no_drop_do_O:
                to_del.append(di)
                continue
            else:
                for tdt in to_del_tag:
                    tags.remove(tdt)

        # df.loc[di, 'ner.text'] = texts
        df.loc[di, 'ner'] = str(tags)
    print(hmap.values())
    if do_concat:
        df.to_csv("new_corpus/{}_{}.csv".format("new_corpus_no_overlap_concat", name),index=False)
    if do_drop:
        df.to_csv("corpus/{}_{}.csv".format("output_test_letr_API", name),index=False)
        # df_del = df.drop(to_del)
        # df_del.to_csv("new_corpus/{}_{}.csv".format("new_corpus_no_overlap_drop", name),index=False)
def train_validation_split(dir = Path("new_corpus/new_corpus_no_overlap.csv"), portion = 0.1):
    import random
    random.seed(portion)
    df = pd.read_csv(dir, sep=',')
    train = list(range(len(df)))

    test = random.sample(train, int(len(df)*portion))
    for i in test: train.remove(i)

    val = random.sample(train, int(len(df)*portion))
    for i in val: train.remove(i)

    print("test : {}\nvalidation : {}\ntrain : {}".format(len(test), len(val), len(train)))

    test_df = pd.DataFrame([df.loc[i] for i in test])
    train_df = pd.DataFrame([df.loc[i] for i in train])
    val_df = pd.DataFrame([df.loc[i] for i in val])
    test_df.to_csv("new_corpus/{}_test_{}.csv".format(dir.name, portion),index=False)
    train_df.to_csv("new_corpus/{}_train_{}.csv".format(dir.name, portion),index=False)
    val_df.to_csv("new_corpus/{}_val_{}.csv".format(dir.name, portion),index=False)

if __name__ == "__main__":
    # corpus_dir = "corpus/NIKL_EL_2021_v1.1/국립국어원 개체명 분석 말뭉치 개체 연결 2021(버전 1.1)"
    # corpus_dir = Path(corpus_dir)
    # json_dir = get_json_list(corpus_dir)
    # # with open("AIHub_new_ner_corpus_221022.json",'w', encoding='utf-8') as file:
    # write_csv(json_dir)

    # find_overlap_token(Path("new_corpus/{}.csv".format("new_corpus_no_overlap")), do_concat=True, name = 'v2')
    # find_overlap_token(Path("corpus/new_corpus_no_overlap.csv_test_0.1.csv_no_special_221028.csv"), do_drop=True, name = "letr", drop_tag_dict = {
    # 'PERSON' : ['PERSON', 'PS_NAME', 'PS_CHARACTER', 'PS_PET'],
    # 'NORP' : ['OGG_RELIGION', 'OGG_POLITICS'],
    # 'FAC' : ['AF_BUILDING', 'AF_CULTURAL_ASSET', 'AF_ROAD',
    #     'AF_TRANSPORT',],
    # 'ORG' : ['OGG_ECONOMY', 'OGG_EDUCATION', 'OGG_MILITARY',
    #     'OGG_MEDIA', 'OGG_SPORTS', 'OGG_ART', 'OGG_MEDICINE',
    #     'OGG_SCIENCE', 'OGG_LIBRARY', 'OGG_LAW',
    #     'OGG_FOOD', 'OGG_HOTEL', 'OGG_OTHERS', 'ORG'],
    # 'GPE' : ['LCP_COUNTRY', 'LCP_PROVINCE', 'LCP_COUNTY',
    #     'LCP_CITY', 'LCP_CAPITALCITY'],
    # 'LOC' : ['LCG_RIVER', 'LCG_OCEAN',
    #     'LCG_BAY', 'LCG_MOUNTAIN', 'LCG_ISLAND',
    #     'LCG_CONTINENT', 'LC_SPACE', 'LC_OTHERS'],
    # 'PRODUCT' :  ['AF_MUSICAL_INSTRUMENT', 'AF_WEAPON',
    #     'AFW_SERVICE_PRODUCTS', 'AFW_OTHER_PRODUCTS', 'PRODUCT',
    #     'PT_FRUIT', 'PT_FLOWER', 'PT_TREE', 'PT_GRASS', 'PT_TYPE',
    #     'PT_PART', 'PT_OTHERS'],
    # 'EVENT' : ['EV_ACTIVITY', 'EV_WAR_REVOLUTION', 'EV_SPORTS',
    #     'EV_FESTIVAL', 'EV_OTHERS'],
    # 'WORK_OF_ART' : ['AFA_DOCUMENT', 'AFA_PERFORMANCE',
    #     'AFA_VIDEO', 'AFA_ART_CRAFT', 'AFA_MUSIC', 'WORK_OF_ART'],
    # 'LAW' : ['CV_POLICY', 'CV_LAW', 'CV_CURRENCY', 'CV_TAX', 'CV_FUNDS'], 
    # 'LANGUAGE' :  ['CV_LANGUAGE'],
    # 'DATE' : ['DT_DURATION', 'DT_DAY', 'DT_WEEK', 'DT_MONTH',
    #     'DT_YEAR', 'DT_SEASON', 'DT_GEOAGE', 'DT_DYNASTY',
    #     'DT_OTHERS'],
    # 'TIME' : ['TI_DURATION', 'TI_HOUR', 'TI_MINUTE', 'TI_SECOND',
    #     'TI_OTHERS'],
    # 'PERCENT' : ['QT_PERCENTAGE'],
    # 'MONEY' : ['QT_PRICE'],
    # 'QUANTITY' : ['QT_AGE', 'QT_SIZE', 'QT_LENGTH', 'QT_COUNT',
    #     'QT_MAN_COUNT', 'QT_WEIGHT',
    #     'QT_SPEED', 'QT_TEMPERATURE', 'QT_VOLUME', 'QT_ORDER',
    #     'QT_PHONE', 'QT_SPORTS', 'QT_CHANNEL',
    #     'QT_ALBUM', 'QT_ADDRESS', 'QT_OTHERS'],
    # 'ORDINAL' : ['QT_ORDER']
    # }, no_drop_do_O = True, concat_tag_dict = {})

    # with open('new_corpus/newcorpus_text.txt', 'w', encoding='utf-8') as file:
    #     df = pd.read_csv('new_corpus/new_corpus_no_overlap_drop_xlmr.csv', sep = ',')
    #     for d in df['ko_original']:
    #         file.write("{}\n".format(re.sub(r"[^\uAC00-\uD7A3a-zA-Z\s.,?]", "", d).strip()))
    # print('done')

    # train_validation_split(portion = 0.1)
    # find_overlap_token(Path("new_corpus/new_corpus_no_overlap.csv"), do_concat=True)

    # dir = Path("output_test_letr_API.csv")
    # df = pd.read_csv(dir, sep = ',')
    # hmap = {}
    # for di, d in tqdm(enumerate(df['ner'])):
    #     tags = (eval(d))
    #     texts = df['sentence'][di]
    #     for ti, tag in enumerate(tags):
    #         t = tag['entity']
    #         try:
    #             if hmap[hash(t)][0] == t:
    #                 hmap[hash(t)][1] += 1
    #                 continue
    #         except:
    #             hmap[hash(t)] = [t, 1]
    # print(hmap.values())
    
    # 괄호 제거
    # dir = Path("corpus/new_corpus_no_overlap.csv_train_0.1.csv")
    # df = pd.read_csv(dir, sep = ',')
    # for i, d in enumerate(df['ko_original']):
    #     ko = d
    #     new_tags = eval(str(df['ner.tags'][i]))
    #     if ko[0] == "> " or ko[0] == " >":
    #         for j, tags in enumerate(eval(str(df['ner.tags'][i]))):
    #             new_tags[j]['position'] = [i-2 for i in eval(str(tags['position']))]
    #             print(tags)
    #             print(new_tags)
    #         ko = ko[2:]
    #     if ko[0] == ">":
    #         for j, tags in enumerate(eval(str(df['ner.tags'][i]))):
    #             new_tags[j]['position'] = [i-1 for i in eval(str(tags['position']))]
    #         ko = ko[1:]
    #     if ko[0] == " ":
    #         for j, tags in enumerate(eval(str(df['ner.tags'][i]))):
    #             new_tags[j]['position'] = [i-1 for i in eval(str(tags['position']))]
    #         ko = ko[1:]
    #     gh = []
    #     double_gh_n = 0 #괄호가 두 개 이상 있을 경우 인덱스 감소
    #     for k, letter in enumerate(ko): #괄호 제거
    #         if letter == '(':
    #             gh.append(k-double_gh_n)
    #         if letter == ')':
    #             gh.append(k-double_gh_n)
    #             if len(gh) == 1: # )하나
    #                 print("-------------------------------")
    #                 print(ko)
    #                 for l, t in enumerate(new_tags):
    #                     # print(t['position'])
    #                     new_tags[l]['position'] = [int(li)-1 if int(li) > gh[0] else int(li) for li in eval(str(t['position']))]
    #                 ko = ko[:gh[0]] + ko[gh[0]+1:]
    #                 gh = []
    #                 print("-------------------------------")
    #             if len(gh) == 2:
    #                 # print(ko[gh[0]+1:gh[1]])
    #                 cut = True
    #                 for tag in new_tags:
    #                     if ko[gh[0]+1:gh[1]] in tag['value']:
    #                         print('in-----')
    #                         print(ko[gh[0]+1:gh[1]], tag['value'])
    #                         print(tag)
    #                         cut = False
    #                         # if tag['tag'] == 'PERSON' or tag['tag'][:3] == 'PS_':
    #                 if cut:
    #                     print('not in-----')
    #                     print(ko[gh[0]+1:gh[1]])
    #                     print(ko)
    #                     ko = ko.replace(ko[gh[0]:gh[1]+1], "")
    #                     print(new_tags)
    #                     for l, t in enumerate(new_tags):
    #                         # print(t['position'])
    #                         new_tags[l]['position'] = [int(li)-len(ko[gh[0]:gh[1]+1]) if int(li) > gh[1] else int(li) for li in eval(str(t['position'])) ]
    #                         # print(len(ko[gh[0]:gh[1]+1]))
    #                     print(ko)
    #                     print(new_tags)
    #                     double_gh_n += len(ko[gh[0]:gh[1]+1])
    #                 gh = []
    #     if len(gh) == 1: # )하나
    #         for l, t in enumerate(new_tags):
    #             # print(t['position'])
    #             new_tags[l]['position'] = [int(li)-1 if int(li) > gh[0] else int(li) for li in eval(str(t['position']))]
    #         ko = ko[:gh[0]] + ko[gh[0]+1:]
    #         gh = []
    #     df.loc[i, 'ko_original'] = ko
    #     df.loc[i, 'ner.tags'] = str(new_tags)
    #     if ko[0] == " ":
    #         for j, tags in enumerate(eval(str(df['ner.tags'][i]))):
    #             new_tags[j]['position'] = [i-1 for i in eval(str(eval(str(df['ner.tags'][i]))[j]['position']))]
    #         ko = ko[1:]
    #         df.loc[i, 'ko_original'] = ko
    #         df.loc[i, 'ner.tags'] = str(new_tags)
    # df.to_csv("corpus/{}_no_special_221028.csv".format(dir.name), sep=',')

    find_overlap_token(Path("corpus/new_corpus_no_overlap.csv_train_0.1.csv"), do_drop = True, drop_tag_dict = {
    'PER' : ['PERSON', 'PS'],
    'ORG' : ['OGG', 'ORG'],
    'PRD' : ['AFW', 'PRODUCT'],
    'WOA' : ['AFA', 'WORK_OF_ART']
    }, name = 'train_data_4_1109')
    find_overlap_token(Path("corpus/new_corpus_no_overlap.csv_val_0.1.csv"), do_drop = True, drop_tag_dict = {
    'PER' : ['PERSON', 'PS'],
    'ORG' : ['OGG', 'ORG'],
    'PRD' : ['AFW', 'PRODUCT'],
    'WOA' : ['AFA', 'WORK_OF_ART']
    }, name = 'val_data_4_1109')
    find_overlap_token(Path("corpus/new_corpus_no_overlap.csv_test_0.1.csv"), do_drop = True, drop_tag_dict = {
    'PER' : ['PERSON', 'PS'],
    'ORG' : ['OGG', 'ORG'],
    'PRD' : ['AFW', 'PRODUCT'],
    'WOA' : ['AFA', 'WORK_OF_ART']
    }, name = 'test_data_4_1109')
    to_train_bert(Path("corpus/new_corpus_no_overlap_no_drop_train_data_4.csv"), 'train')
    to_train_bert(Path("corpus/new_corpus_no_overlap_no_drop_val_data_4.csv"), 'val')
            # print(df['ner.tags'][i])
        # df.loc[i, 'ko_original'] = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9~,.?\s]", "", d).strip()
    # df['w/special'] = with_special
    # df.to_csv('corpus/new_corpus_no_overlap.csv_test_0.1_no_special.csv', sep=',')
    # print('done')
