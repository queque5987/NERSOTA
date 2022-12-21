from transformers import BertTokenizer, BertForMaskedLM, BertConfig, Trainer, TrainingArguments
import torch
import argparse
import os
import json
import gc
def train(kwargs):
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

    def collate_fn(text):
        max_length = kwargs.max_length
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

    if not kwargs.on_memory:
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

    if kwargs.on_memory:
        print("training with dataset on memory\nTHIS PROCESS IS NOT RECOMENDED IF YOU DON'T HAVE MORE THAN 10GB MEMORY AT LEAST")
        if not os.path.isdir("./dataset"):
            os.mkdir("./dataset")
        if not os.path.isfile("./dataset/train_dataset_kcbert-base.pt"):
            print("downloading train_dataset . . .")
            gdownload(file_id = '1XPpZDUgPW6ju8X9XhJZIyTdWpjZNTXQ-', output_name = './dataset/train_dataset_kcbert-base.pt')
        if not os.path.isfile("./dataset/eval_dataset_kcbert-base.pt"):
            print("downloading eval_dataset . . .")
            gdownload(file_id = '1Sk-toHPIPD1e5Y1OBTITo--gBHwf1mfi', output_name = './dataset/eval_dataset_kcbert-base.pt')
        print('loading train_dataset . . .')
        train_dataset = torch.load("dataset/train_dataset_kcbert-base.pt")
        print('loading eval_dataset . . .')
        eval_dataset = torch.load("dataset/eval_dataset_kcbert-base.pt")

    print("start training . . .")

    args = TrainingArguments(
        output_dir="./" + kwargs.model_name,
        per_device_train_batch_size=kwargs.batch_size,
        per_device_eval_batch_size=kwargs.batch_size,
        num_train_epochs=kwargs.epochs,
        evaluation_strategy="steps",
        eval_steps=50000,
        save_strategy="steps",
        save_steps=300000,
        load_best_model_at_end = True
    )

    if kwargs.on_memory:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
    else:
        trainer = Trainer(
            model=model,
            args=args,
            data_collator=collate_fn,
            train_dataset=train_data,
            eval_dataset=eval_data,
        )

    gc.collect()
    torch.cuda.empty_cache()
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='사용법 테스트입니다.')

    parser.add_argument('-d', '--model_name', required=False, default="NERSOTA_BERT", help='Name of model - checkpoint will be saved in this dir')
    parser.add_argument('-e', '--epochs', required=False, default=20, type=int, help='Epochs')
    parser.add_argument('-b', '--batch_size', required=False, default=32, type=int, help='Batch_size')
    parser.add_argument('--eval_steps', required=False, default=50000, type=int, help='Perform evaluation per n steps')
    parser.add_argument('--save_steps', required=False, default=300000, type=int, help='Saves checkpoint of lowest eval_loss in recent eval steps - save_steps = eval_steps * n')
    parser.add_argument('--max_length', required=False, default=64, type=int, help='Tokenizer\'s maximum padding length')
    parser.add_argument('--on_memory', required=False, default=False, help='Loading datasets on memory directly - Requires more than 10GB memory at least')

    args = parser.parse_args()

    train(args)