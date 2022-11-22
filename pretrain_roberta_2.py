from tqdm import tqdm
from transformers import RobertaTokenizer,RobertaConfig, RobertaForMaskedLM, Trainer, TrainingArguments, BatchEncoding
import torch
import json
# from tokenizers import ByteLevelBPETokenizer
import tokenizers

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# tokenizer = ByteLevelBPETokenizer(
#     'bpe/vocab.json',
#     'bpe/merges.txt'
# )
config = RobertaConfig.from_pretrained("roberta-base")
model = RobertaForMaskedLM(config)

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

def get_dataset(text: list, max_length = 128, mode = "normal"):
    if mode == "normal":
        inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    elif mode == "bpe":
        print("tokenizing data")
        inputs = [tokenizer.encode(t) for t in text]
        print("switching data to batchencoding")
        print(inputs[0].ids)
        input_ids = []
        attention_mask = []
        for input in inputs:
            input_ids.append((input.ids+[1 for _ in range(max_length-len(input.ids))]) if len(input.ids) <= max_length else input.ids[:max_length])
            attention_mask.append((input.attention_mask + [0 for _ in range(max_length-len(input.attention_mask))]) if len(input.attention_mask) <= max_length else input.attention_mask[:max_length])
        inputs = BatchEncoding({"input_ids" : torch.tensor(input_ids), "attention_mask" : torch.tensor(attention_mask)})
        # inputs = BatchEncoding({"input_ids" : torch.tensor([(input.ids + [1 for _ in range(max_length-len(input.ids))]) if len(input.ids) <= max_length else input.ids[:max_length]] for input in inputs), #.to(device)
        # "attention_mask" : torch.tensor(
        #     [(input.attention_mask + [0 for _ in range(max_length-len(input.attention_mask))]) if len(input.attention_mask) <= max_length 
        #     else input.attention_mask[:max_length]]
        #         for input in inputs),#.to(device)
        ### ValueError: expected sequence of length 64 at dim 1 (got 69) 2211181303 len(input.ids >> len(input.attention_mask 2211181335 max_length 64 >> 128 since bpe len(text) != len(bpe(text))
        ### RuntimeError: [enfore fail at CPUAllocator.cpp:68] . DefaultCPUAllocator: cant'allocate memory: you tried to allocate 5861894400 bytes. Error code 12 (Cannot allocate memory)
        # "labels" : torch.tensor([input.ids + [1 for _ in range(max_length-len(input.ids))] for input in inputs])
        # })
    inputs['labels'] = inputs.input_ids.detach().clone()#.to(device)
    # create random array of floats with equal dimensions to input_ids tensor
    rand = torch.rand(inputs.input_ids.shape)#.to(device)
    # create mask array
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
            (inputs.input_ids != 102) * (inputs.input_ids != 0)
    # print(mask_arr)
    selection = []
    print("selecting tokens to shift")
    for i in range(inputs.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )

    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103
    print("initializing tensors to dataset")
    dataset = MeditationsDataset(inputs)
    print("done making dataset")
    return dataset

def collate_fn(text):
    max_length = 128
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

train_dir = 'nersota_corpus_for_pretrain_no_special_len_under64_2211141659.json_train_0.05.json'
eval_dir = 'nersota_corpus_for_pretrain_no_special_len_under64_2211141659.json_eval_0.05.json'
mode = "bpe"
print("loading data")
with open(train_dir, 'r', encoding='utf-8') as j_file:
    train_data = json.load(j_file)
with open(eval_dir, 'r', encoding='utf-8') as j_file:
    eval_data = json.load(j_file)
print(train_data[0])
print("gathering dataset")
# train_dataset = get_dataset(train_data, mode=mode)
# train_dataset = get_dataset(train_data[:len(train_data)//2], mode=mode)
# eval_dataset = get_dataset(eval_data, mode=mode)

# from torch.utils.data import DataLoader
# batch_size = 8
# train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
# eval_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)

args = TrainingArguments(
    output_dir='out_roberta_base_pretrained_bpe_20221122',
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=5,
    evaluation_strategy="steps",
    eval_steps=10000,
    save_strategy="steps",
    save_steps=100000,
    # load_best_model_at_end = True,
    # device=device
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=collate_fn,
    train_dataset=train_data,
    eval_dataset=eval_data,
    # place_model_on_device=True
)
# # import gc

# # gc.collect()
# # torch.cuda.empty_cache()

# with torch.no_grad():
trainer.train()
# trainer.fit