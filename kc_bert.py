import torch
class inference():
    def __init__(self, ckpt_dir, label_map_dir):
        import torch
        class Arguments():
            def __init__(self):
                self.pretrained_model_name="beomi/kcbert-base"
                self.downstream_corpus_name="ner"
                self.downstream_model_dir="/nlpbook/checkpoint-ner"
                self.batch_size=32 if torch.cuda.is_available() else 4
                self.learning_rate=5e-5
                self.max_seq_length=64
                self.epochs=3
                self.tpu_cores=0 if torch.cuda.is_available() else 8
                self.seed=7
                self.downstream_corpus_root_dir=""
                self.cpu_workers=0
        self.args = Arguments()
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            self.args.pretrained_model_name,
            do_lower_case=False,
        )
        fine_tuned_model_ckpt = torch.load(
            ckpt_dir,
            map_location = torch.device('cpu')
        )
        from transformers import BertConfig
        pretrained_model_config = BertConfig.from_pretrained(
            self.args.pretrained_model_name,
            num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel()
        )
        from transformers import BertForTokenClassification
        self.model = BertForTokenClassification(pretrained_model_config)
        self.model.load_state_dict({k.replace("model.",""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
        self.device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        labels = [label.strip() for label in open(label_map_dir,'r').readlines()]
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
            outputs = self.model(**{k: torch.tensor(v).to(self.device) for k, v in inputs.items()})
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