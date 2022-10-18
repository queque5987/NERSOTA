import os
import pathlib
import json
from tqdm import tqdm 

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
    
    with open("xlm_entities_to_analyse.txt", "w", encoding= "utf-8") as file:
        # file = json.dump(entity_hmap, file)
        for entity in entity_hmap.values():
            file.write("{}\n-----\n".format(entity))
    with open("xlm_entities_to_analyse_diff.txt", "w", encoding= "utf-8") as file:
        for dat in temp_entity:
            file.write("{}\n{}\n".format(dat[0], dat[1]))
    print('done')