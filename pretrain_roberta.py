from transformers import RobertaConfig, RobertaForMaskedLM, Trainer, TrainingArguments, BatchEncoding, AutoTokenizer
import torch
from tokenizers import ByteLevelBPETokenizer
import argparse
import os
import json
import gc

def train(kwargs):
    if kwargs.use_trained_tokenizer:
        tokenizer = ByteLevelBPETokenizer(
            'bpe/vocab.json',
            'bpe/merges.txt'
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta')
    config = RobertaConfig.from_pretrained("roberta-base")
    model = RobertaForMaskedLM(config)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    def collate_fn(text):
        max_length = kwargs.max_length
        if kwargs.use_trained_tokenizer:
            inputs = [tokenizer.encode(t) for t in text]
            input_ids = []
            attention_mask = []
            for input in inputs:
                input_ids.append((input.ids+[1 for _ in range(max_length-len(input.ids))]) if len(input.ids) <= max_length else input.ids[:max_length])
                attention_mask.append((input.attention_mask + [0 for _ in range(max_length-len(input.attention_mask))]) if len(input.attention_mask) <= max_length else input.attention_mask[:max_length])
            inputs = BatchEncoding({"input_ids" : torch.tensor(input_ids), "attention_mask" : torch.tensor(attention_mask)})
        else: 
            inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
        inputs['labels'] = inputs.input_ids.detach().clone()
        rand = torch.rand(inputs.input_ids.shape)
        mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
                (inputs.input_ids != 102) * (inputs.input_ids != 0)
        selection = []
        for i in range(inputs.input_ids.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )
        for i in range(inputs.input_ids.shape[0]):
            inputs.input_ids[i, selection[i]] = 103
        return inputs
    
    def gdownload(file_id, output_name):
        import gdown
        google_path = 'https://drive.google.com/uc?id='
        gdown.download(google_path+file_id,output_name,quiet=False)

    print("loading datasets . . .")
    if not os.path.isdir("./dataset"):
        os.mkdir("./dataset")
    if not os.path.isfile("./dataset/pretrain_train.json"):
        print("downloading train_dataset . . .")
        gdownload(file_id = '1l4MSKZLAimj2qgSlEUl-soLfPpvrhHtm', output_name = './dataset/pretrain_train.json')
    else:
        print("train_dataset already exists")
    if not os.path.isfile("./dataset/pretrain_eval.json"):
        print("downloading eval_dataset . . .")
        gdownload(file_id = '1CWGHQ1gtQ49dM-hBVPVcv9WnNoCjsu5q', output_name = './dataset/pretrain_eval.json')
    else:
        print("eval_dataset already exists")
    train_dir = 'dataset/pretrain_train.json'
    eval_dir = 'dataset/pretrain_eval.json'
    print("loading datasets . . .")
    with open(train_dir, 'r', encoding='utf-8') as j_file:
        train_data = json.load(j_file)
    with open(eval_dir, 'r', encoding='utf-8') as j_file:
        eval_data = json.load(j_file)
    print(train_data[0])
    print("dataset loaded")

    args = TrainingArguments(
        output_dir="./" + kwargs.model_name,
        per_device_train_batch_size=kwargs.batch_size,
        per_device_eval_batch_size=kwargs.batch_size,
        num_train_epochs=kwargs.epochs,
        evaluation_strategy="steps",
        eval_steps=kwargs.eval_steps,
        save_strategy="steps",
        save_steps=kwargs.save_steps
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        train_dataset=train_data,
        eval_dataset=eval_data,
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training NERSOTA_RoBERTa')

    parser.add_argument('t', '--use_trained_tokenizer', required=False, default=True, help='True if using NERSOTA_tokenizer False if using normal tokenizer')
    parser.add_argument('-d', '--model_name', required=False, default="NERSOTA_RoBERTa", help='Name of model - checkpoint will be saved in this dir')
    parser.add_argument('-e', '--epochs', required=False, default=20, type=int, help='Epochs')
    parser.add_argument('-b', '--batch_size', required=False, default=32, type=int, help='Batch_size')
    parser.add_argument('--eval_steps', required=False, default=50000, type=int, help='Perform evaluation per n steps')
    parser.add_argument('--save_steps', required=False, default=300000, type=int, help='Saves checkpoint of lowest eval_loss in recent eval steps - save_steps = eval_steps * n')
    parser.add_argument('--max_length', required=False, default=64, type=int, help='Tokenizer\'s maximum padding length')

    args = parser.parse_args()

    train(args)