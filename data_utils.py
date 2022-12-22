import gdown
import json
from tqdm import tqdm
def gdownload(file_id, output_name):
    google_path = 'https://drive.google.com/uc?id='
    gdown.download(google_path+file_id,output_name,quiet=False)
def add_sequence_label(output):
    """
        [{'token': '세', 'predicted_tag': 'B-PER', 'top_prob': '0.9992'},
        {'token': '##찬', 'predicted_tag': 'I-PER', 'top_prob': '0.9994'},
        {'token': '##이', 'predicted_tag': 'I-PER', 'top_prob': '0.9557'},
        {'token': '##인가', 'predicted_tag': 'O', 'top_prob': '0.9939'},
        {'token': '봐', 'predicted_tag': 'O', 'top_prob': '0.9999'},
        {'token': '.', 'predicted_tag': 'O', 'top_prob': '0.9999'}]
        >>
        "output": [
            "B-PER",
            "I-PER",
            "I-PER",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O",
            "O"
        ]
    """
    label_map = [
    'O',
    'B-PER',
    'B-CVL',
    'B-DAT',
    'B-QTT',
    'B-THR',
    'B-ANM',
    'B-ORG',
    'B-TRM',
    'B-STF',
    'B-ARF',
    'B-LOC',
    'B-TIM',
    'B-MAT',
    'B-PLT',
    'B-EVT',
    'I-PER',
    'I-CVL',
    'I-DAT',
    'I-QTT',
    'I-THR',
    'I-ANM',
    'I-ORG',
    'I-TRM',
    'I-STF',
    'I-ARF',
    'I-LOC',
    'I-TIM',
    'I-MAT',
    'I-PLT',
    'I-EVT',
    ]
    ndf = []
    def replace_list(new_sentence, token, predicted_tag):
        flag = False
        for token_idx, t in enumerate(token):
            for i, s in enumerate(new_sentence):
                if t == s:
                    if token_idx > 0 and predicted_tag[0] == "B":
                        new_sentence[i] = "I" + predicted_tag[1:]
                    else: new_sentence[i] = predicted_tag
                    flag = True
                    break
        # if not flag:
        #     print("error ", new_sentence, token)
        return new_sentence, flag
    sentence = output['sentence']
    result = output['result']
    new_sentence = [s for s in sentence]
    for i, re in enumerate(result): 
        token = re['token']
        predicted_tag = re['predicted_tag']
        token = token.replace("##", "")
        if token == "[UNK]":
            token = ">"
        new_sentence, flag = replace_list(new_sentence, token, predicted_tag)
    
    for sen_idx, s in enumerate(new_sentence):
        if s not in label_map:
            if s == " ":
                if new_sentence[sen_idx+1][0] == "I":
                    new_sentence[sen_idx] = new_sentence[sen_idx+1]
                else: new_sentence[sen_idx] = "O"
            else:
                new_sentence[sen_idx] = "O"
    output_b = sentence
    n = 0
    for i, out in enumerate(new_sentence):
        if out != "O":
            if out[0] == "B":
                if i+1 >= len(new_sentence) or new_sentence[i+1] == "O":
                    output_b = output_b[:i+n] + "<{}>".format(out[3:]) + output_b[i+n:] + "</{}>".format(out[3:])
                    n += len("<{}>".format(out[3:])) + len("</{}>".format(out[3:]))
                else:
                    output_b = output_b[:i+n] + "<{}>".format(out[3:]) + output_b[i+n:]
                    n += len("<{}>".format(out[3:]))
            elif out[0] == "I":
                output_b = output_b[:i+n] + "</{}>".format(out[3:]) + output_b[i+n:]
                n += len("</{}>".format(out[3:]))

    return {"sentence" : sentence, "output_b" : output_b, "output" : new_sentence, "result" : result}