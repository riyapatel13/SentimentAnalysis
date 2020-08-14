# fastai_sentiment_analysis

This repo contains a sentiment analysis tool created using [fastai's deep learning library](https://docs.fast.ai/) along with [Huggingface transformers](https://huggingface.co/transformers/). Huggingface transformers contain several pretrained NLP models that perform tasks such as tokenization, lemmatization, feature extraction, entity recognition, etc. The current models that can be used for sentiment analysis in this file are:
* [BERT](https://github.com/google-research/bert) (from Google)
* [XLNet](https://github.com/zihangdai/xlnet) (from Google/CMU)
* [XLM](https://github.com/facebookresearch/XLM) (from Facebook)
* [RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta) (from Facebook)
* [DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html) (from HuggingFace)

This code currently uses RoBERTa's pretrained base model and customizes its tokenizer and transformer. It can easily be altered to use any of RoBERTa's other pretrained models, as well as any of the other pretrained models listed above (currently, these are the only models that support multi-class classification). The code uses fastai's library to implement the [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf), which is a transfer learning method that is relatively new to the field of NLP. fastai allows for easy fine-tuning of the last few layers of the deep neural net by calculating discriminative learning rates and performing gradual unfreezing. This method known to provide much more accurate results than other text classification methods.

## Files Included

* exported_models
  * Folder containing the models created when training the data that can be used for inference.
  * ```transformer.pkl``` is a RoBERTa pretrained base model with 3 fine-tuned last layers trained on the Rotten Tomatoes movie review dataset (train.tsv.zip in data folder)
  * ```untrained_learner.pkl``` is a simple RoBERTa pretrained base model 
* fastai_inference.py
  * File to conduct inference. Given a text file with line-separated sentences, this file will predict each sentence's sentiment on a scale from Negative to Positive (Negative, Somewhat negative, Neutral, Somewhat positive, and Positive). The results will be stored in a file passed in by the user. Currently uses ```exported_models/transformer.pkl``` as its model, but can be modified.
* fastai_training.py
  * File to train the model. Given training data as a tsv file, it will train and export a model with fine-tuned layers. Currently uses RoBERTa's pretrained base model, but can be changed to use any of the models above. See file for details on changing models.

