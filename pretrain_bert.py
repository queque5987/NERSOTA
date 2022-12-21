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
config = BertConfig(hidden_size = 768,
                    num_hidden_layers = 12,
                    hidden_act = 'gelu',
                    hidden_dropout_prob = 0.1,
                    attention_probs_dropout_prob = 0.1,
                    max_position_embeddings = 512,
                    layer_norm_eps = 1e-12,
                    # position_embedding_type = 'absolute',
                    classifier_dropout = None)
# config = BertConfig(
#     hidden_act = 'gelu',
#     hidden_size = 1024,
#     num_hidden_layers = 24,
#     num_attention_heads = 16,
#     intermediate_size = 4096,
#     hidden_dropout_prob = 0.1,
# ) #large

# model_file = "./out_kcbert_large_11221658/checkpoint-50000/pytorch_model.bin"
# config_file = "./out_kcbert_large_11221658/checkpoint-50000/config.json"
# optimizer_file = "./out_kcbert_large_11221658/checkpoint-50000/optimizer.pt"


# model = BertForMaskedLM(config)
model = BertForMaskedLM.from_pretrained("./out_kcbert_large_11221658/checkpoint-50000/")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print("model loaded on {}".format(device))

"""
    collate_function define
"""
def collate_fn(text):
    max_length = 16
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

train_dir = 'dataset/nersota_corpus_for_pretrain_no_special_len_under64_2211141659.json_train_0.05.json'
eval_dir = 'dataset/nersota_corpus_for_pretrain_no_special_len_under64_2211141659.json_eval_0.05.json'
mode = "bpe"
print("loading data")
with open(train_dir, 'r', encoding='utf-8') as j_file:
    train_data = json.load(j_file)
with open(eval_dir, 'r', encoding='utf-8') as j_file:
    eval_data = json.load(j_file)
print(type(train_data[0]))
print("gathering dataset")

"""
    Trainer 사용 train
"""

from transformers import TrainingArguments

# args = TrainingArguments(
#     output_dir='out_kcbert_large_11291754',
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=20,
#     evaluation_strategy="steps",
#     eval_steps=10000,
#     save_strategy="steps",
#     save_steps=100000,
#     load_best_model_at_end = True
# )

args = torch.load(".NERtesting/out_kcbert_large_11221658/checkpoint-50000/training_args.bin")

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