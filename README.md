# SentimentAnalysis

This repo contains code to perform sentiment analysis on text documents. This was created as part of an ongoing research project at Carnegie Mellon Human-Computer Interaction Institute under Dr. Nik Martelaro - for more information about the study, visit <b>link</b>. The project is aimed at assisting user researchers in analyzing large amounts of qualitative data that humans are typically tasked with. Specifically, my role was concerned with developing and using off-the-shelf machine learning tools to perform computational analysis on data such as human speech and movements. 

## Files Included

* data_format.py
  * Reads data from input text file (in paragraph form or line-separated sentences) and parses it into separate phrases that can be fed into the sentiment analysis code.
* sentiment_analysis.py
  * This file contains 3 different classifiers to train the data using sci-kit learn. It contains code for a support vector classifier, a random forest classifier, and a multinomial Naive Bayes classifier. It will return a csv file with its predictions.
* ibm_sentiment.py
  * This file contains contains code for the IBM Tone Analyzer. Given a JSON file with the text, it will return a JSON file of tone analysis.
* fastai_sentiment_analysis.py
  * This file contains a sentiment analysis tool using fastai and Huggingface transformers. It currently uses the 'roberta-base' pre-trained model, but it can be configured to use one of these model types: BERT, RoBERTa, XLNet, XLM, DistilBERT. It will return a file with the given predictions in a csv file. 
* Data files
  * train.tsv, train.tsv.zip, test.tsv, test.tsv.zip, driving.rtf
  * The train and test data sets are from a Rotten Tomatoes dataset (from [Kaggle](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data#) Sentiment Analysis competition).
  * The driving file is a transcript of a conversation taken while driving. It is not split by phrases like the other data set.
* fastai model
  * untrain.pth
  * This is the sentiment analysis model created from the roberta pre-trained model.
* stat_parser
  * Folder containing files used used in data_format.py to parse the data into phrases
* sentiment_analysis_results
  * Folder containing results of the sentiment analysis using different models
* sampleSubmission.csv
  * File used in fastai_sentiment_analysis.py to format predictions file

## Installation

In order to run this code, you will need the following: 
* Python (version 3.6 or greater)
* [fastai](https://github.com/fastai/fastai/blob/master/README.md#installation) library 
  ```bash
  pip install fastai
  ```
* [Huggingface transformers](https://github.com/huggingface/transformers#installation) library
  * To install transformers library, you need to install one of, or both, [TensorFlow 2.0](https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available) and [PyTorch](https://pytorch.org/get-started/locally/#start-locally).
  ```bash
  pip install transformers
  ```
  
  ## Running Code
  
  To format the data so that it can properly be fed into the sentiment analysis tools, run
  ```bash
  python data_format.py <input file> <output file>
  ```
  This will save the parsed data into ```<output file>```. 
  
  ## Results
  
  When running the various modes of analysis on different sentences, these are the produced results.
  
  **Sentence:** It's a hoot and a half, and a great way for the American people to see what a candidate is like when he's not giving the same 15-cent stump speech.
  | Method | Result |
  | ------ | ------ |
  | SVM | 2 -> neutral |
  | Random Forest | 3 -> somewhat positive |
  | Multinomial Naive Bayes | 2 -> neutral | 
  | FastAI | 4 -> positive |
  | IBM Watson Tone Analyzer | 46% joy, 93% extraverted, 80% large emotional range  |
  
  **Sentence:** The film is quiet, threatening and unforgettable.
  | Method | Result |
  | ------ | ------ |
  | SVM | 2 -> neutral |
  | Random Forest | 2 -> neutral |
  | Multinomial Naive Bayes | 3 -> somewhat positive | 
  | FastAI | 2 -> neutral |
  | IBM Watson Tone Analyzer | 56% fear, 91% openness, 67% analytical  |
  
  **Sentence:** No one but a convict guilty of some truly heinous crime should have to sit through The Master of Disguise.
  | Method | Result |
  | ------ | ------ |
  | SVM | 1 -> somewhat negative |
  | Random Forest | 1 -> somewhat negative |
  | Multinomial Naive Bayes | 2 -> neutral | 
  | FastAI | 0 -> negative |
  | IBM Watson Tone Analyzer | 38% disgust, 52% confident, 89% openness  |