import torch
from transformers import RobertaTokenizer, AutoTokenizer, BertConfig, BertForTokenClassification
from ratsnlp import nlpbook
from ratsnlp.nlpbook.ner import NERCorpus, NERDataset, NERTrainArguments, NERTask
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import argparse
import pathlib
import os
import data_utils as utils

def train(kwargs):
    print("loading datasets . . .")
    if not os.path.isdir("./{}".format(kwargs.corpus_name)):
        os.mkdir("./{}".format(kwargs.corpus_name))
    if not os.path.isfile("./{}/train.txt".format(kwargs.corpus_name)):
        print("downloading train_dataset . . .")
        utils.gdownload(file_id = '1cx9OMiVK7HJGkJCQkChf4OCCiUTqvbB4', output_name = './{}/train.txt'.format(kwargs.corpus_name))
    else:
        print("train_dataset already exists")
    if not os.path.isfile("./{}/eval.txt".format(kwargs.corpus_name)):
        print("downloading eval_dataset . . .")
        utils.gdownload(file_id = '1R5vfpnmBG6i_pZa2_MFTZII24Rvu4vq0', output_name = './{}/val.txt'.format(kwargs.corpus_name))
    else:
        print("eval_dataset already exists")
    args = NERTrainArguments(
        pretrained_model_name="bert-base-uncased",
        downstream_corpus_name=kwargs.corpus_name,
        downstream_model_dir="./"+kwargs.model_name,
        batch_size=kwargs.batch_size,
        learning_rate=kwargs.learning_rate,
        max_seq_length=kwargs.max_length,
        epochs=kwargs.epochs, 
        tpu_cores=0 if torch.cuda.is_available() else 8,
        seed=kwargs.seed,
        downstream_corpus_root_dir="",
        cpu_workers=0,
        save_top_k = 1
    )
    nlpbook.set_seed(args)
    nlpbook.set_logger(args)
    if kwargs.tokenizer == "NERSOTA" or kwargs.tokenzier == "nersota":
        tokenizer = RobertaTokenizer(
            vocab_file='bpe/vocab.json',
            merges_file='bpe/merges.txt'
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(kwargs.tokenizer)
    corpus = NERCorpus(args)
    train_dataset = NERDataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode="train",
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset, replacement=False),
        collate_fn=nlpbook.data_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
    val_dataset = NERDataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode="val",
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(val_dataset),
        collate_fn=nlpbook.data_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=corpus.num_labels,
    )
    model = BertForTokenClassification.from_pretrained(
            "bert-base-uncased",
            config=pretrained_model_config,
    )
    model.load_state_dict(torch.load(os.path.join(kwargs.checkpoint_dir, "pytorch_model.bin")), strict=False)
    
    task = NERTask(model, args)
    trainer = nlpbook.get_trainer(args)
    trainer.fit(
        task,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training NERSOTA_BERT')

    parser.add_argument('-c', '--checkpoint_dir', required=True, help='Directory where checkpoint files are in ex)pytorch_model.bin')

    parser.add_argument('-d', '--model_name', required=False, default="NERSOTA_RoBERTa", help='Name of model - checkpoint will be saved in this dir')
    parser.add_argument('-t', '--tokenizer', required=False, default="BM-K/KoSimCSE-roberta", help='Tokenizer - \'NERSOTA\' or \'nersota\' to use NERSOTA tokenizer')
    parser.add_argument('-e', '--epochs', required=False, default=20, type=int, help='Epochs')
    parser.add_argument('-b', '--batch_size', required=False, default=32, type=int, help='Batch_size')
    parser.add_argument('-l', '--learning_rate', required=False, default=1e-5, type=int, help='Learning_rate')

    parser.add_argument('--corpus_name', required=False, default="ner", help='Corpus_name')
    parser.add_argument('--seed', required=False, default=14, type=int, help='Random Seed')
    
    parser.add_argument('--eval_steps', required=False, default=50000, type=int, help='Perform evaluation per n steps')
    parser.add_argument('--save_steps', required=False, default=300000, type=int, help='Saves checkpoint of lowest eval_loss in recent eval steps - save_steps = eval_steps * n')
    parser.add_argument('--max_length', required=False, default=64, type=int, help='Tokenizer\'s maximum padding length')

    args = parser.parse_args()

    args.checkpoint_dir = pathlib.Path(args.checkpoint_dir)

    train(args)