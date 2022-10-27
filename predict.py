import torch
# from transformers import BertModel
from tokenization_hanbert import HanBertTokenizer

tokenizer = HanBertTokenizer.from_pretrained('HanBert-54kN-torch')