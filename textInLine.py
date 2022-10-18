import os
import pathlib
import json
from tqdm import tqdm 
import pandas as pd

if __name__ == "__main__":
    file_name = "xlm_inferenced_c_event.txt"
    file_dir = pathlib.Path(file_name)
    org_file_dir = pathlib.Path(file_name.replace('inferenced', 'original'))

    entity_hmap = {}
    temp_entity = []

    with open(file_dir, "r", encoding = 'utf-8') as inf_file:
        with open(org_file_dir, "r", encoding = 'utf-8') as org_file:
            inf = inf_file.readlines()
            org = org_file.readlines()
            for i, _ in enumerate(tqdm(inf)):
                inf_s = inf[i].split()
                org_s = org[i].split()
                if len(inf_s) == len(org_s):
                    for j, _ in enumerate(inf_s):
                        h = hash(org_s[j])
                        try:
                            if entity_hmap[h]:
                                # entity_hmap[h][-1] += 1
                                if inf_s[j] not in entity_hmap[h][1:-1]:
                                    entity_hmap[h].append(inf_s[j])
                                    entity_hmap[h].append(1)
                                else:
                                    # print(entity_hmap[h].index(inf_s[j]))
                                    idx = entity_hmap[h].index(inf_s[j], 1)
                                    entity_hmap[h][idx+1] += 1
                                    # print(inf_s[j])
                                    # print(idx)
                                break
                        except KeyError as e:
                            entity_hmap[h] = [org_s[j], inf_s[j], 1]
                            break
                else:
                    print(inf[i].split())
                    print(org[i].split())
                    temp_entity.append([org[i], inf[i]])

    """
    엑셀 저장
    """
    # xlsx_entities = []
    # for ent in tqdm(entity_hmap.values()):
    #     temp = ent
    #     while(True):
    #         if len(temp) <= 3:
    #             xlsx_entities.append(temp)
    #             break
    #         else:
    #             xlsx_entities.append([temp[0], temp[-2], temp[-1]])
    #             temp = temp[:-2]
    # df = pd.DataFrame(xlsx_entities)
    # df.to_excel("xlm_entities_to_analyse.xlsx", sheet_name= 'new_name', index= False, header= False)

    """
    txt 저장
    """
    # with open("xlm_entities_to_analyse.txt", "w", encoding= "utf-8") as file:
    #     for entity in entity_hmap.values():
    #         file.write("{}\n".format(entity))
    # with open("xlm_entities_to_analyse_diff.txt", "w", encoding= "utf-8") as file:
    #     for dat in temp_entity:
    #         file.write("{}\n{}\n".format(dat[0], dat[1]))
    
    per_entity = []
    for entity in entity_hmap.values():
        s = 0
        d = 0
        original_token = ''
        for i, e in enumerate(entity):
            if i == 0:
                original_token = e
                continue
            if type(e) == int:
                continue
            if e == original_token:
                s += entity[i+1]
            else:
                d += entity[i+1]
        per_entity.append([original_token, d/(s+d)*100, d, s+d])
    
    print(per_entity[:20])
    df = pd.DataFrame(per_entity)
    df.to_excel("xlm_entities_to_analyse_recognized_as_NE_per_total.xlsx", sheet_name= 'new_name', index= False, header= False)

    print('done')