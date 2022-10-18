import os
import pathlib
import json
from tqdm import tqdm
import tfxlmrner40lang as xlm

dat_categories = ["fm_drama", "fs_drama", "enter", "culture", "history"]
txlm = xlm.tfxml()
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
                    index = ddd.find(']')
                    if (dat_category == "fm_drama" or dat_category == "fs_drama") and ddd[:index] == "해설":
                        continue
                    texts.append(ddd[index+1:])
    print(texts[:10])
    with open("xlm_inferenced_{}.txt".format(dat_category), "w", encoding='utf-8') as file:
        for text in tqdm(texts):
            file.write(txlm.inference(text))
            file.write("\n")

print("done")