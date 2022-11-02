import enum
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re

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
                    print(tag)
                    print(output)
                    if j == pos[0]:
                        new_output[j] = "B-{}".format(tag['tag'])
                    else:
                        print(df['ko_original'][i])
                        new_output[j] = "I-{}".format(tag['tag'])
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

def getScores(p_dir, y_dir):
    import numpy as np
    import sklearn.metrics as metrics

    p = pd.read_csv(p_dir)
    y = pd.read_csv(y_dir)
    # p = p['ner'].values.tolist()
    p = p['output'].values.tolist()
    y = y['output'].values.tolist()
    p_new = []
    for pp in p:
        for ppp in eval(pp):
            p_new.append(ppp)
    
    y_new = []
    for pp in y:
        for ppp in eval(pp):
            y_new.append(ppp)
    p = p_new
    y = y_new
    print('model : {}'.format('Spacy'))
    print('accuracy', metrics.accuracy_score(y,p))
    print('precision', metrics.precision_score(y,p,average='micro'))
    print('recall', metrics.recall_score(y,p,average='micro'))
    print('f1 micro', metrics.f1_score(y,p,average='micro'))
    print('f1 macro', metrics.f1_score(y,p,average='macro'))

    print(metrics.classification_report(y,p,zero_division=True))
    # print(metrics.confusion_matrix(y,p))
if __name__ == "__main__":
    # BIO_corpus('corpus/new_corpus_no_overlap_no_drop_letr___.csv', 'corpus/new_corpus_no_overlap_no_drop_letr221028.csv_bio.csv')
    # BIO_xlmr("corpus/output_test_letr_API_no_cardinal.csv", "output_test_letr_API_no_cardinal.csv_renewed.csv")
    # BIO_spacy("output_test_spcay.csv", "output_test_spcay_renewed.csv")
    # BIO_corpus_token("corpus/new_corpus_no_overlap_no_drop_spacy221027.csv", "corpus/new_corpus_no_overlap_no_drop_spacy221027_special_test_0.1_BIO.csv")
    
    getScores("output_test_spcay_renewed.csv", "corpus/new_corpus_no_overlap_no_drop_spacy221027_test_0.1_BIO.csv")
    # getScores("output_test_letr_API_no_cardinal.csv_renewed.csv", "corpus/new_corpus_no_overlap_no_drop_letr221028.csv_bio.csv")
    # getScores("output_test_xlm-roberta-large-finetuned-conll03-english_renewed.csv", 'corpus/new_corpus_no_overlap_no_drop_xlmr_test_0.1_BIO_221026.csv')