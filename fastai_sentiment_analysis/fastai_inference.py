'''
Using this tutorial:
https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta

Modeled after this Kaggle competition:
https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data#
'''
# encoding=utf8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path 

import os
import argparse

import torch
import torch.optim as optim

import random 
import time
import json
tic = time.perf_counter()
# fastai
from fastai import *
from fastai.text import *
from fastai.callbacks import *

# transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig

from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig

# optimizer / metric
from transformers import AdamW
from functools import partial

'''
fastai_inference.py

This code loads a learner trained in fastai_sentiment_analysis.py and strictly 
performs inference. Since a custom model was used, all new classes must be declared
in this file in order for the learner to work correctly. Given a text file with
sentences (separated by new lines), this file will predict each sentence's sentiment 
on a scale from Negative to Positive (Negative, Somewhat negative, Neutral, 
Somewhat positive, and Positive). The results will be stored in a file passed in 
by the user.

Command to run:
python3 fastai_inference.py <input_file> <output_file>
'''

# different architecture dictionaries (for easy access)
MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig)
}
# Select model type
model_type = 'roberta'
pretrained_model_name = 'roberta-base'

model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]



# Custom Tokenizer - inherits BaseTokenizer but overwrites 
class TransformersBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.max_len
        self.model_type = model_type

    def __call__(self, *args, **kwargs): 
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length and add the spesial tokens"""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        if self.model_type in ['roberta']:
            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
            tokens = [CLS] + tokens + [SEP]
        else:
            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
            if self.model_type in ['xlnet']:
                tokens = tokens + [SEP] +  [CLS]
            else:
                tokens = [CLS] + tokens + [SEP]
        return tokens

# Custom Numericalizer
'''
Creating new class that inherits from Vocab class and overwrites numericalize and textify.
'''
class TransformersVocab(Vocab):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos = [])
        self.tokenizer = tokenizer
    
    def numericalize(self, t:Collection[str]) -> List[int]:
        # Convert a list of tokens `t` to their ids.
        return self.tokenizer.convert_tokens_to_ids(t)
        #return self.tokenizer.encode(t)

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        # Convert a list of `nums` to their tokens.
        nums = np.array(nums).tolist()
        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)
    
    def __getstate__(self):
        return {'itos':self.itos, 'tokenizer':self.tokenizer}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.tokenizer = state['tokenizer']
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

'''
According to HuggingFace documentation, BERT, RoBERTa, XLM and DistilBERT are models 
with absolute position embeddings, so it's usually advised to pad the inputs on the right rather than the left.

For XLNET, it is model with relative position embeddings, so the padding can go on either side.
'''
transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
pad_first = bool(model_type in ['xlnet'])
pad_idx = transformer_tokenizer.pad_token_id

# Custom Model 
'''
Creating custom model to access the logits.
'''
# defining our model architecture 
class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel):
        super(CustomTransformerModel,self).__init__()
        self.transformer = transformer_model
        
    def forward(self, input_ids, attention_mask=None):
        
        # attention_mask
        # Mask to avoid performing attention on padding token indices.
        # Mask values selected in ``[0, 1]``:
        # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        attention_mask = (input_ids!=pad_idx).type(input_ids.type()) 
        
        logits = self.transformer(input_ids,
                                  attention_mask = attention_mask)[0]   
        return logits

# Loading learner used for predictions
#export_learner = load_learner('./', file = 'transformer.pkl')
#toc = time.perf_counter()
#print(f"Time taken to import functions and load learner: {toc-tic:0.4f} seconds")

#sentence = input('Type a sentence that you want to be analyzed for sentiment:\n')
#tac = time.perf_counter()
#prediction = export_learner.predict(sentence)

#categories = {0:'Negative', 1:'Somewhat negative', 2:'Neutral', 3:'Somewhat positive', 4:'Postive'}
#print(categories[int(prediction[0])])
#tuc = time.perf_counter()
#print(f"Time taken to predict and print: {tuc-tac:0.4f} seconds")

# Predictions
'''
def get_preds_as_nparray(ds_type) -> np.ndarray:
    """
    the get_preds method does not yield the elements in order by default
    we borrow the code from the RNNLearner to resort the elements into their correct order
    """
    preds = export_learner.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in databunch.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    return preds[reverse_sampler, :]

test_preds = get_preds_as_nparray(DatasetType.Test)
sample_submission = pd.read_csv('sampleSubmission.csv')
sample_submission['Sentiment'] = np.argmax(test_preds,axis=1)
sample_submission.to_csv("car_preds.csv", index=False)
'''
# Python File I/O
def readFile(path):
    with open(path, "rt") as f:
        return f.readlines()

def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)

def predict_file(infile, outfile):
    export_learner = load_learner('./exported_models/', file = 'transformer.pkl')
    categories = {0:'Negative', 1:'Somewhat negative', 2:'Neutral', 3:'Somewhat positive', 4:'Postive'}

    intext = readFile(infile)
    data = {}
    json_list = []

    for line in intext:
        if line == ' ' or line == ' ' or line == '\n' or line == '\t':
            continue
        prediction = export_learner.predict(line)
        data["sentence"] = line
        data["sentiment"] = categories[int(prediction[0])]
        json_dump = json.dumps(data)
        json_list.append(json_dump)

    with open(outfile, 'w') as out:
        out.writelines(["%s\n" % item for item in json_list])

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description = "Sentiment Analysis Inference - Given an input file where each sentence is separated by a new line, it will classify each sentence as either Negative, Somewhat Negative, Neutral, Somewhat Positive, or Positive. It will then save the predictions in a new file. The learner is trained on Rotten Tomatoes movie reviews. In order to train the learner with a different dataset, please refer to the file \"fastai_training.py\".")
    argparser.add_argument("input_file",
                        type=str,
                        help="file containing sentences to be analyzed (line-separated)")
    argparser.add_argument("output_file",
                        type=str,
                        help="file sentiment analysis for each sentence")

    args = argparser.parse_args()


predict_file(args.input_file, args.output_file)
print(f"Prediction complete. The results are saved in a file named {args.output_file}.")
toc = time.perf_counter()
print(f"Prediction time: {toc-tic:0.4f} seconds")
