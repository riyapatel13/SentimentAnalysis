# fastai_sentiment_analysis

This repo contains a sentiment analysis tool created using [fastai's deep learning library](https://docs.fast.ai/) along with [Huggingface transformers](https://huggingface.co/transformers/). Huggingface transformers contain several pretrained NLP models that perform tasks such as tokenization, lemmatization, feature extraction, entity recognition, etc. The current models that can be used for sentiment analysis in this file are:
* [BERT](https://github.com/google-research/bert) (from Google)
* [XLNet](https://github.com/zihangdai/xlnet) (from Google/CMU)
* [XLM](https://github.com/facebookresearch/XLM) (from Facebook)
* [RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta) (from Facebook)
* [DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html) (from HuggingFace)

This code currently uses RoBERTa's pretrained base model and customizes its tokenizer and transformer. It can easily be altered to use any of RoBERTa's other pretrained models, as well as any of the other pretrained models listed above.