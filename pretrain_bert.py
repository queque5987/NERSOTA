from transformers import BertTokenizer, BertForMaskedLM, BertConfig, Trainer
# import json
# from toCSV import get_json_list
from tqdm import tqdm
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
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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
                    position_embedding_type = 'absolute',
                    classifier_dropout = None)
model = BertForMaskedLM(config)

"""
    테스트 1회
"""

# text = scripts[0]

# inputs = tokenizer(text, return_tensors='pt')
# print(inputs.keys())
# print(inputs)

# inputs['labels'] = inputs.input_ids.detach().clone()
# print(inputs)

# # create random array of floats in equal dimension to input_ids
# rand = torch.rand(inputs.input_ids.shape)
# # where the random array is less than 0.15, we set true
# mask_arr = rand < 0.15
# print(mask_arr)
# print((inputs.input_ids != 101) * (inputs.input_ids != 102))
# mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102)
# print(mask_arr)
# selection = torch.flatten((mask_arr[0]).nonzero()).tolist()
# inputs.input_ids[0, selection] = 103
# outputs = model(**inputs)
# print(outputs.keys())
# print(outputs.loss)

# text = scripts[:100000]

"""
    Dataset gathering
"""

class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

# class TokenizingDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings):
#         from transformers import BertTokenizer
#         self.encodings = encodings
#         self.tokenizer = BertTokenizer.from_pretrained(
#             "beomi/kcbert-base",
#             do_lower_case=False,
#         )
#     def __getitem__(self, idx):
#         encodings = self.get_data(self.encodings[idx])
        
#         # print("key : {}, value : {}".format(k, v))
#         # print(type({key: torch.tensor(val) for key, val in encodings.items()}))
#         return encodings[0]
#         return {key: torch.tensor(val) for key, val in encodings.items()}
#     def __len__(self):
#         return len(self.encodings)
#     def get_data(self, text):
#         inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
#         inputs['labels'] = inputs.input_ids.detach().clone()
#         # create random array of floats with equal dimensions to input_ids tensor
#         rand = torch.rand(inputs.input_ids.shape)
#         # create mask array
#         mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
#                 (inputs.input_ids != 102) * (inputs.input_ids != 0)
#         selection = []
#         for i in range(inputs.input_ids.shape[0]):
#             selection.append(
#                 torch.flatten(mask_arr[i].nonzero()).tolist()
#             )
#         for i in range(inputs.input_ids.shape[0]):
#             inputs.input_ids[i, selection[i]] = 103
#         # print(inputs)
#         return inputs

# def tokenize(text : list):
#     inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
#     return inputs

def get_dataset(text : list):
    inputs = tokenizer(text, return_tensors='pt', max_length=64, truncation=True, padding='max_length')
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
    # print(selection[:5])

    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103
    # print(inputs.input_ids)

        
    dataset = MeditationsDataset(inputs)
    return dataset
# name = 'nersota_corpus_for_pretrain_no_special_len_under512_2211141659'
# j = "{}.json".format(name)
# with open(j, 'r', encoding='utf-8') as j_file:
#     train_data = json.load(j_file)
# import random

# random.shuffle(dict)

# portion = 0.1

# test_data = dict[:int(len(dict)*portion)]
# eval_data = dict[int(len(dict)*portion):int(len(dict)*portion)*2]
# train_data = dict[int(len(dict)*portion)*2:]

# with open("{}_{}_test.json".format(name,portion), "w", encoding='utf-8') as outfile:
#     json.dump(test_data, outfile, indent=2, ensure_ascii=False)
# with open("{}_{}_eval.json".format(name,portion), "w", encoding='utf-8') as outfile:
#     json.dump(eval_data, outfile, indent=2, ensure_ascii=False)
# with open("{}_{}_train.json".format(name,portion), "w", encoding='utf-8') as outfile:
#     json.dump(train_data, outfile, indent=2, ensure_ascii=False)

# print("test : {} eval : {} train : {}".format(len(test_data), len(eval_data), len(train_data)))

# with open("corpus/nersota_corpus_for_pretrain_no_special_len_under512_2211141659_0.1_eval.json",'r', encoding='utf-8') as j_file:
#     eval_dict = json.load(j_file)
# eval_datasets = []

# # print(type(eval_dict))
# # print(eval_dict[:10])
# # print(type(eval_dict[:10]))
# iter_eval = list(range(0, len(eval_dict), len(eval_dict)//100))
# iter_eval.append(len(eval_dict)-1)
# # iter_eval = iter_eval[1:]
# for i, itrn in tqdm(enumerate(iter_eval)):
#     if i == 0: continue
#     eval_datasets.append(get_dataset(eval_dict[iter_eval[i-1]:itrn]))
#     # if i >= 67:
#     #     break
#     # if i%10 == 0:
#     #     eval_dataset = torch.utils.data.ConcatDataset(eval_datasets)
#     #     torch.save(eval_dataset, "train_dataset_{}_{}/{}.pt".format(tokenizer_name, itrn, len(eval_dict)))
#     #     eval_datasets = []
# # eval_datasets.append(get_dataset(eval_dict[:2]))
# # # print([d for d in eval_dataset])
# # eval_datasets.append(get_dataset(eval_dict[2:4]))
# eval_dataset = torch.utils.data.ConcatDataset(eval_datasets)
# # # print([d for d in eval_dataset])
# # # for itern in range(1,101):
# # #     itern
# # #     eval_dataset = get_dataset(eval_dict)
# # torch.save(eval_dataset, "eval_dataset_{}.pt".format(tokenizer_name))
# torch.save(eval_dataset, "eval_dataset_{}.pt".format('kcbert-base'))

""" 2022.11.15 """

# import datetime
# print('train_dataset . . .\n{}'.format(datetime.datetime.now()))
# name = 'nersota_corpus_for_pretrain_no_special_len_under64_2211141659.json_train_0.05.json'
# j = "corpus/{}".format(name)
# with open(j, 'r', encoding='utf-8') as j_file:
#     train_data = json.load(j_file)
# train_dataset = get_dataset(train_data)
# torch.save(train_dataset, "corpus/train_dataset_{}.pt".format('kcbert-base'))

# print('eval_dataset . . .\n{}'.format(datetime.datetime.now()))
# name = 'nersota_corpus_for_pretrain_no_special_len_under64_2211141659.json_eval_0.05.json'
# j = "corpus/{}".format(name)
# with open(j, 'r', encoding='utf-8') as j_file:
#     eval_data = json.load(j_file)
# eval_dataset = get_dataset(eval_data)
# torch.save(eval_dataset, "corpus/eval_dataset_{}.pt".format('kcbert-base'))
# print('loaded_dataset\n{}'.format(datetime.datetime.now()))

""" 2022.11.15 """

# name = 'nersota_corpus_for_pretrain_no_special_len_under512_2211141659_0.1_train'
# j = "corpus/{}.json".format(name)
# with open(j, 'r', encoding='utf-8') as j_file:
#     train_data = json.load(j_file)
# train_dataset = TokenizingDataset(train_data)

# name = 'nersota_corpus_for_pretrain_no_special_len_under512_2211141659_0.1_eval'
# j = "corpus/{}.json".format(name)
# with open(j, 'r', encoding='utf-8') as j_file:
#     eval_data = json.load(j_file)
# eval_dataset = TokenizingDataset(eval_data)

"""
    load 테스트
"""

import datetime
print('train_dataset . . .\n{}'.format(datetime.datetime.now()))
train_dataset = torch.load("corpus/train_dataset_kcbert-base.pt")

print('eval_dataset . . .\n{}'.format(datetime.datetime.now()))
eval_dataset = torch.load("corpus/eval_dataset_kcbert-base.pt")
# train3 = torch.load("train_dataset_kcbert-base2.pt")
# print(type(train1))

# with open("nersota_corpus_for_pretrain_no_special_2211141633_eval.json", "w", encoding='utf-8') as outfile:
#     json.dump(eval_data, outfile, indent=2, ensure_ascii=False)
# with open("nersota_corpus_for_pretrain_no_special_2211141633_train.json", "w", encoding='utf-8') as outfile:
#     json.dump(train_data, outfile, indent=2, ensure_ascii=False)

"""
    Dataloader 선언 및 수동 train
"""



# data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
# model.to(device)
# # activate training mode
# model.train()

# from transformers import AdamW
# initialize optimizer
# optim = AdamW(model.parameters(), lr=5e-5)

# from tqdm import tqdm  # for our progress bar

# epochs = 2
# min_loss = 9999
# losses = []
# with torch.no_grad():
#     for epoch in range(epochs):
#         # setup loop with TQDM and dataloader
#         loop = tqdm(data_loader, leave=True)
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
""" loss 감소 안보임 """

# from torch import nn
# from torch.optim import lr_scheduler
# from tqdm import trange
# import os
# from transformers import AdamW

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# def train(parameters):

#     # Train & Eval Dataloader
#     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
#     eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=16, shuffle=True)
#     train_steps = len(train_dataloader)
#     eval_steps = len(eval_dataloader)

#     # Prepare Custom Model
#     model.to(device)   # 모델의 장치를 device 에 할당
#     model.zero_grad()  # 모델 gradient 초기화
#     model.train()      # Train 모드로 모델 설정

#     # Loss Function
#     criterion = nn.CrossEntropyLoss()  # loss function

#     # Optimizer & LR_Scheduler setting
#     optimizer = AdamW(model.parameters(), lr=5e-5)
#     scheduler = lr_scheduler.StepLR(optimizer,  # 선형 스케줄러 세팅 - 학습률 조정용 스케줄러
#                                     parameters['scheduler_step'],
#                                     parameters['scheduler_gamma'])

#     # Train Start
#     train_iterator = trange(int(parameters['epoch']), desc="Epoch")  # 학습 상태 출력을 위한 tqdm.trange 초기 세팅
#     global_step = 0

#     # Epoch 루프
#     for epoch in train_iterator:
#         epoch_iterator = tqdm(
#             train_dataloader, desc='epoch: X/X, global: XXX/XXX, tr_loss: XXX'  # Description 양식 지정
#         )
#         epoch = epoch + 1

#         # Step(batch) 루프
#         for step, batch in enumerate(epoch_iterator):
#             # 모델이 할당된 device 와 동일한 device 에 연산용 텐서 역시 할당 되어 있어야 함
#             # tensor, tags = map(lambda elm: elm.to(device), batch)  # device 에 연산용 텐서 할당
#             out = model(batch)      # Calculate
#             # loss = criterion(out, tags)    # loss 연산
#             loss = out.loss

#             # Backward and optimize
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             scheduler.step()  # Update learning rate schedule
#             global_step += 1
#             # One train step Done

#             # Step Description
#             epoch_iterator.set_description(
#                 'epoch: {}/{}, global: {}/{}, tr_loss: {:.3f}'.format(
#                     epoch, parameters['epoch'],
#                     global_step, train_steps * parameters['epoch'],
#                     loss.item()
#                 )
#             )

#         # -- Evaluate & Save model result -- #
#         # 한 Epoch 종료 시 평가, 평가 결과 정보를 포함한 이름으로 학습된 모델을 지정된 경로에 저장
#         eval_result = evaluate(model, criterion, eval_dataloader)
#         # Set Save Path
#         os.makedirs(parameters['train_output'], exist_ok=True)
#         save_path = parameters['train_output'] + \
#                     f"/epoch-{epoch}-acc-{eval_result['mean_acc']}-loss-{eval_result['mean_loss']}"
#         # Save
#         torch.save(model.state_dict(), save_path + "-model.pth")

# def evaluate(model, criterion, eval_dataloader):
#     # Evaluation
#     model.eval()  # 모델의 AutoGradient 연산을 비활성화하고 평가 연산 모드로 설정 (메모리 사용 및 연산 효율화를 위해)
#     sum_eval_acc, sum_eval_loss = 0, 0
#     eval_result = {"mean_loss": 0, "mean_acc": 0}

#     eval_iterator = tqdm(  # Description 양식 지정
#         eval_dataloader, desc='Evaluating - mean_loss: XXX, mean_acc: XXX'
#     )

#     # Evaluate
#     for e_step, e_batch in enumerate(eval_iterator):
#         # tensor, tags = map(lambda elm: elm.to(device), e_batch)  # device 에 연산용 텐서 할당
#         out = model(e_batch)  # Calculate
#         # loss = criterion(out, tags)
#         loss = out.loss

#         # Calculate acc & loss
#         sum_eval_acc += (out.max(dim=1)[1] == tags).float().mean().item()  # 정답과 추론 값이 일치하는 경우 정답으로 count
#         sum_eval_loss += loss.item()

#         # 평가 결과 업데이트
#         eval_result.update({"mean_loss": sum_eval_acc / (e_step + 1),
#                             "mean_acc": sum_eval_loss / (e_step + 1)})

#         # Step Description
#         eval_iterator.set_description(
#             'Evaluating - mean_loss: {:.3f}, mean_acc: {:.3f}'.format(
#                 eval_result['mean_loss'], eval_result['mean_acc'])
#         )
#     model.train()  # 평가 과정이 모두 종료 된 뒤, 다시 모델을 train 모드로 변경

#     return eval_result
# parameters = {
#         "epoch": 2,                # 전체 학습 Epoch
#         "batch_size": 16,           # Batch Size
#         "learning_rate": 0.005,     # 학습률
#         "scheduler_step": 100,      # 어느정도 step 주기로 학습률을 감소할지 지정
#         "scheduler_gamma": 0.9,     # 학습률의 감소 비율
#         "train_output": "output_221115 1739"    # 학습된 모델의 저장 경로
#     }
# train(parameters)
"""
    Trainer 사용 train
"""

from transformers import TrainingArguments

args = TrainingArguments(
    output_dir='out_kcbert_base_221115_2332',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=20,
    evaluation_strategy="steps",
    eval_steps=50000,
    save_strategy="steps",
    save_steps=300000,
    load_best_model_at_end = True
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
import gc

gc.collect()
torch.cuda.empty_cache()

# with torch.no_grad():
trainer.train()

# from datasets import load_metric
# def compute_metrics(eval_preds):
#     metric = load_metric("glue", "mrpc")
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

# from transformers import TrainingArguments

# args = TrainingArguments(
#     output_dir='out_1115',
#     per_device_train_batch_size=4,
#     num_train_epochs=2,
#     evaluation_strategy="epoch"
# )

# from transformers import Trainer

# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     compute_metrics=compute_metrics,

# )
# import gc

# gc.collect()
# torch.cuda.empty_cache()

# # with torch.no_grad():
# trainer.train()