from transformers import BertTokenizer, BertForMaskedLM, BertConfig, Trainer, BatchEncoding
from tqdm import tqdm
# from tokenizers import BertWordPieceTokenizer
import torch
# import transformers
# from datasets import Dataset, DatasetDict
# from torch.utils.data import DataLoader
import json

# tokenizer = BertWordPieceTokenizer("bwp/vocab.txt", lowercase=False)
# print("tokenizer loaded")
tokenizer_name = "beomi/kcbert-base"
tokenizer = BertTokenizer.from_pretrained(
    tokenizer_name,
    do_lower_case=False,
)
# config = BertConfig(hidden_size = 768,
#                     num_hidden_layers = 12,
#                     hidden_act = 'gelu',
#                     hidden_dropout_prob = 0.1,
#                     attention_probs_dropout_prob = 0.1,
#                     max_position_embeddings = 512,
#                     layer_norm_eps = 1e-12,
#                     # position_embedding_type = 'absolute',
#                     classifier_dropout = None)
config = BertConfig(
    hidden_act = 'gelu',
    hidden_size = 1024,
    num_hidden_layers = 24,
    num_attention_heads = 16,
    intermediate_size = 4096,
    hidden_dropout_prob = 0.1,
) #large
model = BertForMaskedLM(config)

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
device1 = torch.device('cuda:1')
model.to(device)
print("model loaded on {}".format(device))

"""
    collate_function define
"""
def collate_fn(text):
    max_length = 16
    # print("tokenizing data")
    # inputs = [tokenizer.encode(t) for t in text]
    # # inputs = tokenizer.encode(text)
    # # input_ids = (inputs.ids+[1 for _ in range(max_length-len(inputs.ids))]) if len(inputs.ids) <= max_length else inputs.ids[:max_length]
    # # attention_mask = (inputs.attention_mask + [0 for _ in range(max_length-len(inputs.attention_mask))]) if len(inputs.attention_mask) <= max_length else inputs.attention_mask[:max_length]
    # input_ids = []
    # attention_mask = []
    # for input in inputs:
    #     input_ids.append((input.ids+[1 for _ in range(max_length-len(input.ids))]) if len(input.ids) <= max_length else input.ids[:max_length])
    #     attention_mask.append((input.attention_mask + [0 for _ in range(max_length-len(input.attention_mask))]) if len(input.attention_mask) <= max_length else input.attention_mask[:max_length])
    # inputs = BatchEncoding({"input_ids" : torch.tensor(input_ids), "attention_mask" : torch.tensor(attention_mask)})
    inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')#.to(device1)
    inputs['labels'] = inputs.input_ids.detach().clone()#.to(device1)
    # create random array of floats with equal dimensions to input_ids tensor
    rand = torch.rand(inputs.input_ids.shape)#.to(device1)
    # create mask array
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
            (inputs.input_ids != 102) * (inputs.input_ids != 0)
    # print(mask_arr)
    selection = []
    # print("selecting tokens to shift")
    for i in range(inputs.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )

    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103
    # print("initializing tensors to dataset")
    # dataset = MeditationsDataset(inputs)
    # print("done making dataset")
    return inputs

train_dir = 'nersota_corpus_for_pretrain_no_special_len_under32_2211141659_train_0.05.json'
eval_dir = 'nersota_corpus_for_pretrain_no_special_len_under32_2211141659_eval_0.05.json'
mode = "bpe"
print("loading data")
with open(train_dir, 'r', encoding='utf-8') as j_file:
    train_data = json.load(j_file)
with open(eval_dir, 'r', encoding='utf-8') as j_file:
    eval_data = json.load(j_file)
print(type(train_data[0]))
print("gathering dataset")

"""
    Dataset gathering
"""

# class MeditationsDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings
#     def __getitem__(self, idx):
#         return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#     def __len__(self):
#         return len(self.encodings.input_ids)

# def get_dataset(text: list, max_length = 96, mode = "normal"):
#     if mode == "normal":
#         inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
#     elif mode == "bpe":
#         inputs = [tokenizer.encode(t) for t in text[:len(text)//2]]
#         inputs = BatchEncoding({"input_ids" : torch.tensor([input.ids + [1 for _ in range(max_length-len(input.ids))] for input in inputs]),#.to(device),
#         "attention_mask" : torch.tensor([input.attention_mask + [0 for _ in range(max_length-len(input.attention_mask))] for input in inputs]),#.to(device),
#         })
#     inputs['labels'] = inputs.input_ids.detach().clone()#.to(device)
#     # create random array of floats with equal dimensions to input_ids tensor
#     rand = torch.rand(inputs.input_ids.shape)#.to(device)
#     # create mask array
#     mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
#             (inputs.input_ids != 102) * (inputs.input_ids != 0)
#     # print(mask_arr)
#     selection = []

#     for i in range(inputs.input_ids.shape[0]):
#         selection.append(
#             torch.flatten(mask_arr[i].nonzero()).tolist()
#         )

#     for i in range(inputs.input_ids.shape[0]):
#         inputs.input_ids[i, selection[i]] = 103
        
#     dataset = MeditationsDataset(inputs)
#     return dataset

# print("loading datasets")
# train_dir = 'nersota_corpus_for_pretrain_no_special_len_under64_2211141659.json_train_0.05.json'
# eval_dir = 'nersota_corpus_for_pretrain_no_special_len_under64_2211141659.json_eval_0.05.json'
# mode = "bpe"
# with open(train_dir, 'r', encoding='utf-8') as j_file:
#     train_data = json.load(j_file)
# print("train data loaded")
# with open(eval_dir, 'r', encoding='utf-8') as j_file:
#     eval_data = json.load(j_file)
# print("eval data loaded")

# train_dataset = get_dataset(train_data, mode=mode)
# print("train dataset loaded")
# eval_dataset = get_dataset(eval_data, mode=mode)
# print("eval dataset loaded")
"""
    Trainer 사용 train
"""

from transformers import TrainingArguments

args = TrainingArguments(
    output_dir='out_kcbert_large_11201443',
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=80,
    evaluation_strategy="steps",
    eval_steps=100000,
    save_strategy="steps",
    save_steps=1000000,
    load_best_model_at_end = True
)

from transformers import Trainer
trainer = Trainer(
    model=model,
    args=args,
    data_collator=collate_fn,
    train_dataset=train_data,
    eval_dataset=eval_data,
    # place_model_on_device=True
)
# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset
# )

# import gc
# gc.collect()
# torch.cuda.empty_cache()

trainer.train()