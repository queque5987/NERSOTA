import os
import pathlib
import json
from tqdm import tqdm
import tfxlmrner40lang as xlm

def dataset_xlm():
    dat_categories = ["c_event", "fm_drama", "fs_drama", "enter", "culture", "history"]
    # dat_categories = ["fm_drama", "fs_drama", "enter", "culture", "history"]
    # txlm = xlm.tfxml()
    for dat_category in dat_categories:
        print("processing --- {}".format(dat_category))
        data_dir = pathlib.Path("방송대본요약/1.Training/원천데이터/TS1/{}".format(dat_category))
        data = os.scandir(data_dir)
        texts = []
        for dat in data:
            print(dat)
            cate_dir = os.path.join(data_dir, dat)
            for subcate in os.scandir(dat):
                with open(subcate, "r", encoding = 'utf-8') as file:
                    d_json = json.load(file)
                    dddd = d_json['Meta']['passage'].strip().split("\n")
                    for ddd in dddd:
                        while(True):
                            index = ddd.find(']')
                            if (dat_category == "fm_drama" or dat_category == "fs_drama") and ddd[:index] == "해설":
                                continue
                            g_index = ddd.find('(')
                            ge_index = ddd.find(')')

                        texts.append(ddd[index+1:])
        print(texts[:10])
        with open("xlm_original_{}.txt".format(dat_category), "w", encoding='utf-8') as file:
            for text in tqdm(texts):
                # file.write(txlm.inference(text))
                file.write(text)
                file.write("\n")

    print("done")

def dataset_NIKL_NEv1():
    file_name = "SXNE1902007240.json"
    data_dir = pathlib.Path(file_name)
    with open(data_dir, "r", encoding = 'utf-8') as file:
        d_json = json.load(file)
        doc_json = d_json['document']
        categories = []
        for doc in doc_json:
            categories.append(re.sub(r"[^\uAC00-\uD7A3a-zA-Z\s]", "", doc['metadata']['title']))
        categories.sort()
        pprint(categories)

    print("done")

if __name__ == "__main__":
    dataset_xlm()
    # dataset_NIKL_NEv1()
