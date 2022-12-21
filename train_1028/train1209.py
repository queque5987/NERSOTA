import torch
from ratsnlp.nlpbook.ner import NERTrainArguments
class kcbert_inference():
    def __init__(self):
        import torch
        self.args = NERTrainArguments(
            pretrained_model_name="beomi/kcbert-base",
            downstream_corpus_name="ner",
            downstream_model_dir="workspace/nlpbook/mo_kcbert_original",
            batch_size=32,# if torch.cuda.is_available() else 4,
            learning_rate=5e-5,
            max_seq_length=64,
            epochs=7, 
            tpu_cores=0 if torch.cuda.is_available() else 8,
            seed=7,
            downstream_corpus_root_dir="",
            cpu_workers=0,
            save_top_k = 3
        )
        from ratsnlp import nlpbook
        nlpbook.set_seed(self.args)
        nlpbook.set_logger(self.args)
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "beomi/kcbert-base",
            do_lower_case=False,
        )
        import torch
        fine_tuned_model_ckpt = torch.load(
            './workspace/nlpbook/kcbert_original_15/epoch=2-val_loss=0.15.ckpt',
        )
        from transformers import BertConfig
        pretrained_model_config = BertConfig.from_pretrained(
            self.args.pretrained_model_name,
            num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel()
        )
        from transformers import BertForTokenClassification
        self.model = BertForTokenClassification(pretrained_model_config)
        self.model.load_state_dict({k.replace("model.",""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
        self.model.eval()
        labels = [label.strip() for label in open('./workspace/nlpbook/kcbert_original_15/label_map.txt','r').readlines()]
        self.id_to_label = {}
        for idx, label in enumerate(labels):
            self.id_to_label[idx] = label
    def inference_fn(self, sentence):
        inputs = self.tokenizer(
            [sentence],
            max_length = 64,
            padding = 'max_length',
            truncation = True
        )
        with torch.no_grad():
            outputs = self.model(**{k: torch.tensor(v) for k, v in inputs.items()})
            probs = outputs.logits[0].softmax(dim=1)
            top_probs, preds = torch.topk(probs, dim=1, k=1)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            predicted_tags = [self.id_to_label[pred.item()] for pred in preds]
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
class inference():
    def __init__(self):
        import torch
        self.args = NERTrainArguments(
            pretrained_model_name="bert-base-uncased",
            downstream_corpus_name="ner_mo",
            downstream_model_dir="/nlpbook/bert-base475_5e5",
            batch_size=32 if torch.cuda.is_available() else 4,
            learning_rate=5e-5,
            max_seq_length=64,
            epochs=3, 
            tpu_cores=0 if torch.cuda.is_available() else 8,
            seed=7,
            downstream_corpus_root_dir="",
            cpu_workers=0
        )
        from ratsnlp import nlpbook
        nlpbook.set_seed(self.args)
        nlpbook.set_logger(self.args)
        # nlpbook.download_downstream_dataset(args)
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "beomi/kcbert-base",
            do_lower_case=False,
        )
        import torch
        ckpt_dir = "./workspace/nlpbook/bert_base_t_kcbert_15/nlpbook/checkpoint-ner/epoch=4-val_loss=0.18.ckpt"
        print(ckpt_dir)
        fine_tuned_model_ckpt = torch.load(
            ckpt_dir,
        )
        from transformers import BertConfig
        pretrained_model_config = BertConfig.from_pretrained(
            self.args.pretrained_model_name,
            num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel()
        )
        from transformers import BertForTokenClassification
        self.model = BertForTokenClassification(pretrained_model_config)
        self.model.load_state_dict({k.replace("model.",""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
        self.model.eval()
        labels = [label.strip() for label in open('./workspace/nlpbook/bert_base_t_kcbert_15/nlpbook/checkpoint-ner/label_map.txt','r').readlines()]
        self.id_to_label = {}
        for idx, label in enumerate(labels):
            self.id_to_label[idx] = label
    def inference_fn(self, sentence):
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
            predicted_tags = [self.id_to_label[pred.item()] for pred in preds]
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
def train_kc():
    args = NERTrainArguments(
        pretrained_model_name="beomi/kcbert-base",
        downstream_corpus_name="ner_mo_s",
        downstream_model_dir="workspace/nlpbook/mos_kcbert_original",
        batch_size=32,# if torch.cuda.is_available() else 4,
        learning_rate=1e-5,
        max_seq_length=64,
        epochs=3, 
        tpu_cores=0 if torch.cuda.is_available() else 8,
        seed=7,
        downstream_corpus_root_dir="",
        cpu_workers=0,
        save_top_k = 1
    )
    from ratsnlp import nlpbook
    nlpbook.set_seed(args)
    nlpbook.set_logger(args)
    # nlpbook.download_downstream_dataset(args)
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name,
        do_lower_case=False,
    )
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
    from transformers import BertConfig
    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=corpus.num_labels,
    )
    from transformers import BertForTokenClassification
    model = BertForTokenClassification.from_pretrained(
            args.pretrained_model_name,
            config=pretrained_model_config,
    )
    from ratsnlp.nlpbook.ner import NERTask
    task = NERTask(model, args)
    trainer = nlpbook.get_trainer(args)
    trainer.fit(
        task,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
def train():
    import torch
    assert torch.cuda.is_available() == True
    print(torch.cuda.is_available())
    args = NERTrainArguments(
        pretrained_model_name="bert-base-uncased",
        downstream_corpus_name="ner_mo_s",
        downstream_model_dir="workspace/nlpbook/mos_bert-base475_1e5",
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
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(
        "beomi/kcbert-base",
        do_lower_case=False,
    )
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
    from transformers import BertConfig
    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=corpus.num_labels,
    )
    from transformers import BertForTokenClassification
    import json
    ckpt_dir = "checkpoint-4750000_bert/"
    model = BertForTokenClassification.from_pretrained(
            "bert-base-uncased",
            config=pretrained_model_config,
    )
    model.load_state_dict(torch.load(ckpt_dir + "pytorch_model.bin"), strict=False)
    from ratsnlp.nlpbook.ner import NERTask
    task = NERTask(model, args)
    trainer = nlpbook.get_trainer(args)
    trainer.fit(
        task,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )

if __name__ == "__main__":
    # inf_model = inference()
    # print(inf_model.inference_fn("미국은 영국으로부터 식민지배를 받았다. 남북전쟁으로 흑인이 노예에서 해방되었고, 1997년 내가 태어났다."))
    train_kc()