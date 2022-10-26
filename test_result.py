import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def BIO_corpus(file_dir, result_file_dir):
    result = Path(file_dir)
    df = pd.read_csv(result)
    new_df = {'ko_original' : [], 'output' : []}
    for i, outputs in tqdm(enumerate(df['ner.tags'])): #['ko_original', 'ner.tags'...]
        ko_original = df['ko_original'][i]
        new_output = ['O' for _ in range(len(ko_original))]
        
        output = eval(outputs)
        for tag in output:
            pos = eval(str(tag['position']))
            if tag['tag'] != 'O':
                for j in range(int(pos[0]), int(pos[1])):
                    if j == pos[0]:
                        new_output[j] = "B-{}".format(tag['tag'])
                    else: new_output[j] = "I-{}".format(tag['tag'])
        new_df['ko_original'].append(ko_original)
        new_df['output'].append(new_output)
    new_df = pd.DataFrame(new_df)
    new_df.to_csv('{}'.format(result_file_dir))

def BIO_xlmr(file_dir, result_file_dir):
    result = Path(file_dir)
    df = pd.read_csv(result)
    
    for i, outputs in tqdm(enumerate(df['output'])): #['ko_original', 'output']
        ko_original = df['ko_original'][i]
        new_output = ['O' for _ in range(len(ko_original))]

        output = eval(outputs)
        # if len(output) > 0:
        for tag in output: #['entity', 'score', 'index', 'word', 'start', 'end']
            for j in range(tag['start'],tag['end']):
                if j == tag['start']:
                    new_output[j] = "B-{}".format(tag['entity'])
                else: new_output[j] = "I-{}".format(tag['entity'])
        df.loc[i, 'output'] = new_output
    df.to_csv('{}'.format(result_file_dir))
    print('done')

def getScores(p_dir, y_dir):
    import numpy as np
    import sklearn.metrics as metrics

    p = pd.read_csv(p_dir)
    y = pd.read_csv(y_dir)
    p = p['output'].values.tolist()
    y = y['output'].values.tolist()

    print('accuracy', metrics.accuracy_score(y,p))
    print('precision', metrics.precision_score(y,p,average='micro'))
    print('recall', metrics.recall_score(y,p,average='micro'))
    print('f1', metrics.f1_score(y,p,average='micro'))

    # print(metrics.classification_report(y,p))
    # print(metrics.confusion_matrix(y,p))
if __name__ == "__main__":
    # BIO_corpus('corpus/new_corpus_no_overlap_no_drop_xlmr221026.csv', 'corpus/new_corpus_no_overlap_no_drop_xlmr_test_0.1_BIO_221026.csv')
    getScores('output_test_xlmr.csv_renewed.csv', 'corpus/new_corpus_no_overlap_no_drop_xlmr_test_0.1_BIO_221026.csv')