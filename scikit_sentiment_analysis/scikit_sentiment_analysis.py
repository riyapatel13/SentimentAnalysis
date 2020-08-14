import pandas as pd
import spacy
import argparse
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

'''
scikit_sentiment_analysis.py

This file contains three different classifiers to train the data set:
    1.) Support Vector Classifier
    2.) Random Forest Classifier
    3.) Multinomial Naive Bayes

These are implemented using Sci-kit Learn.
'''

# file paths
training_data_path = "../data/train.tsv"
test_data_path = "../data/test.tsv"


'''
Given the path to the Rotten Tomatoes movie review training dataset, the function
will return the training data in 2 lists, where X_train is the list of sentences
and Y_train is the sentiment assigned to those sentences. For any index i, 
the sentence X_train[i] has the sentiment Y_train[i].
'''
def import_training_data(file_path = "../data/test.tsv"):
    source = open(file_path, 'r').readlines()
    # get rid of title
    source = source[1:]

    X_train = []
    Y_train = []

    for line in source:
        # puts phrase id, sentence id, sentence, assignment in list
        line_elts = line.split('\t')[2:]
        # gets rid of newline in last element (assignment)
        line_elts[1] = int(line_elts[1][:-1])
        X_train.append(line_elts[0])
        Y_train.append(line_elts[1])
    
    return X_train, Y_train

'''
Given the path to the Rotten Tomatoes movie review test dataset, the function
will return the test data in a list (a list of sentences).
'''
def import_test_data(file_path = "../data/test.tsv"):
    source = open(file_path, 'r').readlines()
    # get rid of title
    source = source[1:]

    X_test = []

    for line in source:
        # puts phrase id, sentence id, sentence, assignment in list
        line_elts = line.split('\t')[2:]
        # gets rid of newline in last element (assignment)
        X_test.append(line_elts[0][:-1]) 

    return X_test

'''
Performs tokenization of sentences. This function builds a dictionary of features
and transforms each sentence in the training data into a feature vector and saves
the results into X_train_counts.
'''
def vectorize_sentences(X_train):
    count_vect = CountVectorizer()
    # potentially what is causing the poor prediction because the sentences might not be vectorizing well since they are complex
    X_train_counts = count_vect.fit_transform(X_train) 
    return count_vect, X_train_counts

# SVM Classifier
def run_svm():
    X_train, Y_train = import_training_data(training_data_path)
    X_test = import_test_data(test_data_path)
    count_vect, X_train_counts = vectorize_sentences(X_train)

    # downscale weights for words that occur in many documents in the corpus and are therefore less informative
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)

    # setting up classifier and fitting
    # changing the values of C and gamma will also vary the results
    clf = svm.SVC(gamma=0.1, C=300)
    clf.fit(X_train_tf,Y_train)

    # prediction
    predicted = clf.predict(X_test)

    # writing to file - return X_test, predicted. Take in file name
    svm_res = open("sentiment_analysis_results/driving_results_svm.tsv", "w")
    for doc, category in zip(X_test, predicted):
        svm_res.write('%r => %s\n' % (doc, predicted[category]))
    svm_res.close()

# Multinomial Naive Bayes Classifier
def run_nb():
    X_train, Y_train = import_training_data(training_data_path)
    X_test = import_test_data(test_data_path)
    count_vect, X_train_counts = vectorize_sentences(X_train)
    
    # downscale weights for words that occur in many documents in the corpus and are therefore less informative
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    docs_new = X_test
    clf = MultinomialNB().fit(X_train_tfidf, Y_train)

    # create vectors for test data
    test_counts = count_vect.transform(docs_new)
    test_tfidf = tfidf_transformer.transform(test_counts)

    # prediction
    predicted = clf.predict(test_tfidf)

    
    # submission file
    new_file = open("sentiment_analysis_results/driving_results_NB.tsv", "w")
    for doc, category in zip(X_test, predicted):
        new_file.write('%r => %s\n' % (doc, predicted[category])) #was Y_train
    new_file.close()
    
# Random Forest Classifier
def run_rand_forest():
    X_train, Y_train = import_training_data(training_data_path)
    X_test = import_test_data(test_data_path)
    count_vect, X_train_counts = vectorize_sentences(X_train)

    # changing the number of estimators will also vary the results
    text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    text_classifier.fit(X_train_counts, Y_train)
    test_vec = count_vect.transform(X_test)

    # prediction
    predicted = text_classifier.predict(test_vec)

    # writing to file - return X_test, predicted. Take in file name
    print("test",predicted)
    count_res = open("sentiment_analysis_results/driving_results_randomforest.tsv", "w")
    for doc, category in zip(X_test, predicted):
        count_res.write('%r => %s\n' % (doc, predicted[category]))
    count_res.close()
    

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description = "Scikit Sentiment Analysis - Given an input file and the algorithm type, this file will predict the sentiment of the sentences in the input file by training the data and using the algorithm given. The data it trains on is the train.tsv (data from Rotten Tomato movie reviews). It will classify each output on a scale of 0 to 4, where 0 => Negative, 1 => Somewhat Negative, 2 => Neutral, 3 => Somewhat Positive, and 4 => Positive. The results will be saved in the output file name provided.")
    argparser.add_argument("input_file",
                        type=str,
                        help="file containing sentences to be analyzed (line-separated)")
    argparser.add_argument("algorithm",
                        type=str,
                        help="algorithm to train data. select nb (Multi-nomial Naive Bayes), svm (Support Vector Classifier), or rf (Random Forest)")
    argparser.add_argument("output_file",
                        type=str,
                        help="file sentiment analysis for each sentence")
    
    args = argparser.parse_args()

    if args.algorithm == 'nb':
        pass
    elif args.algorithm == 'svm':
        pass
    elif args.algorithm == 'rf':
        pass
    else:
        print("Please re-run with one of these algorithms: nb (Multi-nomial Naive Bayes), svm (Support Vector Classifier), or rf (Random Forest)")
