from transformers import BertTokenizer, BertForMaskedLM, BertConfig, Trainer
# import json
# from toCSV import get_json_list
# from tqdm import tqdm
# import torch

# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# config = BertConfig(hidden_size = 768,
#                     num_hidden_layers = 12,
#                     hidden_act = 'gelu',
#                     hidden_dropout_prob = 0.1,
#                     attention_probs_dropout_prob = 0.1,
#                     max_position_embeddings = 512,
#                     layer_norm_eps = 1e-12,
#                     position_embedding_type = 'absolute',
#                     classifier_dropout = None)
# model = BertForMaskedLM(config)

# corpus_list = get_json_list('pretrain_corpus1108')
# corpus_texts = []
# for corpus in tqdm(corpus_list):
#     with open(corpus, "r", encoding = 'utf-8') as f:
#         j = json.load(f)
#         print(corpus, len(j.get('scripts')))
#         corpus_texts += j.get('scripts')
# print(len(corpus_texts))
# # inputs = bert_tokenizer(corpus_texts[0], return_tensors='pt')
# # inputs['labels'] = inputs.input_ids.detach().clone()
# # # print(inputs)
# # rand = torch.rand(inputs.input_ids.shape)
# # mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) # 101, 102번 토큰 제외하고 15% 위치 선별

# # selection = torch.flatten((mask_arr[0]).nonzero())
# # print(selection)
# trainer = ReformerTrainer(dataset, model, tokenizer,model_name=config.model_name, checkpoint_path=config.checkpoint_path,max_len=config.max_seq_len, train_batch_size=config.batch_size,
#                             eval_batch_size=config.batch_size)

# train_dataloader, eval_dataloader = trainer.build_dataloaders(train_test_split=0.1)

# trainer.train(epochs=config.epochs,
#                 train_dataloader=train_dataloader,
#                 eval_dataloader=eval_dataloader,
#                 log_steps=config.log_steps,
#                 ckpt_steps=config.ckpt_steps,
#                 gradient_accumulation_steps=config.gradient_accumulation_steps)

import torch
import transformers
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

# from torchsampler import ImbalancedDatasetSampler
import json

j = 'nersota_corpus_for_pretrain.json'

with open(j, 'r', encoding='utf-8') as j_file:
    dict = json.load(j_file)
# print(dict['metadata'])
scripts = dict['scripts']
# dataset = Dataset.from_dict({'text' : dict['scripts']}, split=['train[:80%]', 'test[-20%:]'])
# # train_10_80pct_ds = datasets.load_dataset('bookcorpus')
# print(dataset)

from transformers import BertTokenizer, BertForMaskedLM
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig(hidden_size = 768,
                    num_hidden_layers = 12,
                    hidden_act = 'gelu',
                    hidden_dropout_prob = 0.1,
                    attention_probs_dropout_prob = 0.1,
                    max_position_embeddings = 512,
                    layer_norm_eps = 1e-12,
                    position_embedding_type = 'absolute',
                    classifier_dropout = None)
model = BertForMaskedLM(config)

text = scripts[0]

inputs = tokenizer(text, return_tensors='pt')
print(inputs.keys())
print(inputs)

inputs['labels'] = inputs.input_ids.detach().clone()
print(inputs)

# create random array of floats in equal dimension to input_ids
rand = torch.rand(inputs.input_ids.shape)
# where the random array is less than 0.15, we set true
mask_arr = rand < 0.15
print(mask_arr)
print((inputs.input_ids != 101) * (inputs.input_ids != 102))
mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102)
print(mask_arr)
selection = torch.flatten((mask_arr[0]).nonzero()).tolist()
inputs.input_ids[0, selection] = 103
outputs = model(**inputs)
print(outputs.keys())
print(outputs.loss)

text = scripts[:100000]

inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
# print(inputs)
inputs['labels'] = inputs.input_ids.detach().clone()
# print(inputs.keys())
# create random array of floats with equal dimensions to input_ids tensor
rand = torch.rand(inputs.input_ids.shape)
# create mask array
mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
           (inputs.input_ids != 102) * (inputs.input_ids != 0)
# print(mask_arr)
selection = []

for i in range(inputs.input_ids.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )
print(selection[:5])

for i in range(inputs.input_ids.shape[0]):
    inputs.input_ids[i, selection[i]] = 103
# print(inputs.input_ids)

class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)
    
dataset = MeditationsDataset(inputs)

loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
model.to(device)
# activate training mode
model.train()

# from transformers import AdamW
# # initialize optimizer
# optim = AdamW(model.parameters(), lr=5e-5)

# from tqdm import tqdm  # for our progress bar

# epochs = 10
# min_loss = 9999
# losses = []
# with torch.no_grad():
#     for epoch in range(epochs):
#         # setup loop with TQDM and dataloader
#         loop = tqdm(loader, leave=True)
#         for batch in loop:
#             # initialize calculated gradients (from prev step)
#             optim.zero_grad()
#             # pull all tensor batches required for training
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
#             # process
#             outputs = model(input_ids, attention_mask=attention_mask,
#                             labels=labels)
#             # extract loss
#             loss = outputs.loss
#             # calculate loss for every parameter that needs grad update
#             loss.requires_grad_(True) # to fix : RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
#             loss.backward()
#             # update parameters
#             optim.step()
#             # print relevant info to progress bar
#             loop.set_description(f'Epoch {epoch}')
#             loop.set_postfix(loss=loss.item())
#             # print(type(loss.item()))
#             # print(loss.item())
#             min_loss = loss.item() if loss.item() < min_loss else min_loss
#         print("min_loss : {}".format(min_loss))
#         losses.append({epoch : min_loss})
#         min_loss = 9999
# print(losses)

from transformers import TrainingArguments

args = TrainingArguments(
    output_dir='out',
    per_device_train_batch_size=4,
    num_train_epochs=2
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset
)
import gc

gc.collect()
torch.cuda.empty_cache()

# with torch.no_grad():
trainer.train()