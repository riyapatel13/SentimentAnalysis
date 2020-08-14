# SentimentAnalysis

This repo contains code to perform sentiment analysis on text documents. This was created as part of an ongoing research project at Carnegie Mellon Human-Computer Interaction Institute under Dr. Nik Martelaro - for more information about the study, visit <b>link</b>. The project is aimed at assisting user researchers in analyzing large amounts of qualitative data that humans are typically tasked with. Specifically, my role was concerned with developing and using off-the-shelf machine learning tools to perform computational analysis on data such as human speech and movements. 

## Files Included
Please see the READMEs in each folder for more detailed information.

* data 
  * Repo containing data files used for training and testing models.
* parsing_files
  * Repo containing code to parse sentences and create more training data.
* scikit_sentiment_analysis
  * Repo containing code for 3 different classifiers to train the data using [Scikit-learn](https://scikit-learn.org/stable/). It contains code for a support vector classifier, a random forest classifier, and a multinomial Naive Bayes classifier.
* ibm_sentiment_analysis
  * Repo containing code for sentiment analysis of sentences using the [IBM Watson Tone Analyzer](https://www.ibm.com/cloud/watson-tone-analyzer).
* fastai_sentiment_analysis
  * Repo containing a sentiment analysis tool using [fastai library](https://docs.fast.ai/) and [Huggingface transformers](https://huggingface.co/transformers/). It currently uses the 'roberta-base' pre-trained model, but it can be configured to use one of these model types: BERT, RoBERTa, XLNet, XLM, DistilBERT. 
* README.md
* requirements.txt
  * Python packages that need to be installed


## Installation

In order to run this code, create a [virtual environment](https://docs.python.org/3/library/venv.html) and run the following in your virtual environment:
```bash
  pip install -r requirements.txt
```
This will install all packages and configure necessary dependencies needed for the code. For more information about running the files, see the READMEs located in each folder.


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