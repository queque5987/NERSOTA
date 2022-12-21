from ratsnlp.nlpbook.ner import NERTrainArguments
import torch
class roberta():
    def __init__(self):
        import torch
        self.args = NERTrainArguments(
            pretrained_model_name="roberta-base",
            downstream_corpus_name="ner_pure",
            downstream_model_dir="workspace/nlpbook/roberta_t",
            batch_size=16 if torch.cuda.is_available() else 4,
            learning_rate=5e-5,
            max_seq_length=128,
            epochs=3, 
            tpu_cores=0 if torch.cuda.is_available() else 8,
            seed=7,
            downstream_corpus_root_dir="",
            cpu_workers=0
        )
        from ratsnlp import nlpbook
        # nlpbook.set_seed(self.args)
        # nlpbook.set_logger(self.args)
        # nlpbook.download_downstream_dataset(args)
        # from transformers import AutoTokenizer
        # from tokenizers import ByteLevelBPETokenizer
        # tokenizer = ByteLevelBPETokenizer(
        #     'bpe/vocab.json',
        #     'bpe/merges.txt'
        # )
        # from transformers import PreTrainedTokenizerFast
        # self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        from transformers import RobertaTokenizer
        self.tokenizer = RobertaTokenizer(vocab_file='bpe/vocab.json', merges_file='bpe/merges.txt')
        fine_tuned_model_ckpt = torch.load(
            # ckpt_dir,
            # 'train_1028/pytorch_model.bin',
            './workspace/nlpbook/roberta_t_trained_BPE_15/epoch=7-val_loss=0.21.ckpt',
            # map_location = torch.device('cpu')
        )
        from transformers import RobertaConfig
        # pretrained_model_config = BertConfig.from_pretrained(
        #     self.args.pretrained_model_name,
        #     num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel()
        # )
        pretrained_model_config = RobertaConfig.from_pretrained(
            self.args.pretrained_model_name,
            num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel()
        )
        # self.model = BertForTokenClassification(pretrained_model_config)
        from transformers import RobertaForTokenClassification
        self.model = RobertaForTokenClassification.from_pretrained(
                "roberta-base",
                config=pretrained_model_config,
        )
        self.model.load_state_dict({k.replace("model.",""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
        self.model.eval()
        labels = [label.strip() for label in open('./workspace/nlpbook/roberta_t_trained_BPE_15/label_map.txt','r').readlines()]
        self.id_to_label = {}
        for idx, label in enumerate(labels):
            self.id_to_label[idx] = label
        
    def inference_fn(self, sentence):
        import torch
        inputs = self.tokenizer(
            [sentence],
            max_length = 128,
            padding = 'max_length',
            truncation = True
        )
        with torch.no_grad():
            outputs = self.model(**{k: torch.tensor(v) for k, v in inputs.items()})
            probs = outputs.logits[0].softmax(dim=1)
            top_probs, preds = torch.topk(probs, dim=1, k=1)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            # print(tokens)
            tokens = [self.tokenizer.convert_tokens_to_string(token) for token in tokens]
            # print(tokens)
            predicted_tags = [self.id_to_label[pred.item()] for pred in preds]
            # print(predicted_tags)
            # print("-----")
            result = []
            for token, predicted_tag, top_prob in zip(tokens, predicted_tags, top_probs):
                if token not in [self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token]:
                    token_result = {
                        "token" : token,
                        "predicted_tag" : predicted_tag,
                        "top_prob" : str(round(top_prob[0].item(), 4)),
                    }
                    result.append(token_result)
        return {
            "sentence" : sentence,
            "result" : result
        }

def train():
    import torch
    assert torch.cuda.is_available() == True
    print(torch.cuda.is_available())
    args = NERTrainArguments(
        pretrained_model_name="roberta-base",
        downstream_corpus_name="ner_mo_s",
        downstream_model_dir="workspace/nlpbook/mos_roberta_t",
        batch_size=32,# if torch.cuda.is_available() else 4,
        learning_rate=1e-5,
        max_seq_length=128,
        epochs=5, 
        tpu_cores=0 if torch.cuda.is_available() else 8,
        seed=14,
        downstream_corpus_root_dir="",
        cpu_workers=0,
        save_top_k = 1
    )
    from ratsnlp import nlpbook
    nlpbook.set_seed(args)
    nlpbook.set_logger(args)
    # nlpbook.download_downstream_dataset(args)
    # from transformers import AutoTokenizer
    # from tokenizers import ByteLevelBPETokenizer
    # tokenizer = ByteLevelBPETokenizer(
    #     'bpe/vocab.json',
    #     'bpe/merges.txt'
    # )
    # from transformers import PreTrainedTokenizerFast
    # tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer(vocab_file='bpe/vocab.json', merges_file='bpe/merges.txt')

    from ratsnlp.nlpbook.ner import NERCorpus, NERDataset
    from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
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
    from transformers import RobertaConfig
    pretrained_model_config = RobertaConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=corpus.num_labels,
    )
    from transformers import RobertaForTokenClassification
    import json
    ckpt_dir = "./checkpoint-2300000_roberta_NERSOTA/"
    model = RobertaForTokenClassification.from_pretrained(
            "roberta-base",
            config=pretrained_model_config,
    )
    model.load_state_dict(torch.load(ckpt_dir + "pytorch_model.bin"), strict=False)
    from ratsnlp.nlpbook.ner import NERTask
    task = NERTask(model, args)
    trainer = nlpbook.get_trainer(args)
    trainer.fit(
        task,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

if __name__ == "__main__":
    # inf_model = inference()
    # print(inf_model.inference_fn("미국은 영국으로부터 식민지배를 받았다. 남북전쟁으로 흑인이 노예에서 해방되었고, 1997년 내가 태어났다."))
    train()