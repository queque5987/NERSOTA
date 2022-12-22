import torch
from transformers import BertTokenizer, AutoTokenizer, RobertaTokenizer, BertForTokenClassification, BertConfig, RobertaConfig, RobertaForTokenClassification
import argparse
import pathlib
import os
import data_utils as utils
class inference():
    def __init__(self, kwargs):
        self.tokenizer = BertTokenizer.from_pretrained(
            kwargs.tokenizer,
            do_lower_case=False,
        )
        fine_tuned_model_ckpt = torch.load(
            kwargs.checkpoint_dir,
        )
        if kwargs.model.lower() == "roberta":
            pretrained_model_config = RobertaConfig.from_pretrained(
            self.args.pretrained_model_name,
            num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel()
            )
            self.model = RobertaForTokenClassification.from_pretrained(
                "roberta-base",
                config=pretrained_model_config,
            )
        else:
            pretrained_model_config = BertConfig.from_pretrained(
                "bert-base-uncased",
                num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel()
            )
            self.model = BertForTokenClassification(pretrained_model_config)
        
        self.model.load_state_dict({k.replace("model.",""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
        self.model.eval()
        labels = [label.strip() for label in open('./workspace/nlpbook/bert_base_t_kcbert_15/nlpbook/checkpoint-ner/label_map.txt','r').readlines()]
        self.id_to_label = {}
        for idx, label in enumerate(labels):
            self.id_to_label[idx] = label
        self.max_length = kwargs.max_length
    
    def inference_fn(self, sentence):
        inputs = self.tokenizer(
            [sentence],
            max_length = self.max_length,
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training NERSOTA_BERT')

    parser.add_argument('-m', '--model', required=True, help='BERT or RoBERTa')
    parser.add_argument('-c', '--checkpoint_dir', required=True, help='Directory where checkpoint files are in ex)\'./NERSOTA_BERT/epoch=1-val_loss=0.18.ckpt\'')
    parser.add_argument('--text', required=False, default="유재석은 대한민국의 개그맨이다.", type=str, help='Text to inference')

    parser.add_argument('-l', '--load_as_file', required=False, default="", type=str, help='.Json to inference as a file')
    parser.add_argument('-s', '--save_dir', required=False, default="./output", help='Result will be saved in this directory')

    parser.add_argument('-d', '--model_name', required=False, default="NERSOTA_RoBERTa", help='Name of model - checkpoint will be saved in this dir')
    parser.add_argument('-t', '--tokenizer', required=False, default="beomi/kcbert-base", help='Tokenizer - \'NERSOTA\' or \'nersota\' to use NERSOTA tokenizer don\'t set if you are using BERT')
    parser.add_argument('--max_length', required=False, default=128, type=int, help='Tokenizer\'s maximum padding length')

    args = parser.parse_args()

    model = inference(args)

    args.checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    if args.load_as_file != "":
        import json
        args.load_as_file = pathlib.Path(args.load_as_file)
        args.save_dir = pathlib.Path(args.save_dir)
        with open(args.load_as_file, "r", encoding="utf-8") as f:
            lines = json.load(f)
            output = []
            for line in lines:
                output.append(utils.add_sequence_label(model.inference_fn(line)))
        with open(args.save_dir, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    else:
        print(utils.add_sequence_label(model.inference_fn(args.text)))