# sci-kit_sentiment_analysis

This repo contains code to perform sentiment analysis on text documents using [Scikit-learn](https://scikit-learn.org/stable/). This has been implemented using 3 different supervised classification methods:
1. Support Vector Classifier
2. Naive Bayes Classifier
3. Random Forest Classifier

## Files Included

* sentiment_analysis.py
  * This file contains 3 different classifiers to train the data using sci-kit learn. It contains code for the support vector classifier, the random forest classifier, and the multinomial Naive Bayes classifier. Currently, it trains each classifier using the Rotten Tomatoes movie review dataset (train.tsv in data folder, originally from [Kaggle](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data#) competition). It trains the data and returns a csv file with its predictions.
* sentiment_analysis_results
  * This contains the predicition results of running each classifier on driving.txt. Each prediction is formatted as ```sentence => prediction```, where the prediction ranges from 0 - 4 (0 => Negative, 1 => Somewhat Negative, 2 => Neutral, 3 => Somewhat Positive, and 4 => Positive).

## Running Code

To train the data using a specific classifier and view the results, run
  ```bash
  python3 scikit_sentiment_analysis.py <input file> <classifier type> <output file>
  ```
* ```<input file>``` is the file containing line-separated sentences that need to be analyzed
* ```<classifier type>``` specifies which classifier will be trained - "nb" (multinomial Naive Bayes), "svm" (Support Vector Classifier), or "rf" (Random Forest)
* ```<output file>``` contain the results of the prediction in the format ```sentence => prediction```
