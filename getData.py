import os
import pathlib
import json

data_dir = pathlib.Path("방송대본요약/1.Training/원천데이터/TS1/enter")
data = os.scandir(data_dir)
for dat in data:
    # print(dat)
    # cate_dir = os.path.join(data_dir, dat)
    for subcate in os.scandir(dat):
        # for subcate in os.scandir(category):
            print("----------------------------")
            print(subcate.name)
            with open(subcate, "r", encoding = 'utf-8') as file:
                d_json = json.load(file)
                # print(d_json['Meta']['passage'].strip())
                dddd = d_json['Meta']['passage'].strip().split("\n")
                for ddd in dddd:
                    print(ddd)
# print(dddd.strip())
# for dd in dddd.split("\n"):
#     print(dd)