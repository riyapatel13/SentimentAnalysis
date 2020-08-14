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

# Python File I/O
def readFile(path):
    with open(path, "rt").readlines() as f:
        return f.read()

def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)


training_data_path = "../data/train.tsv"

source = open(training_data_path, 'r').readlines()
# get rid of title
source = source[1:]

X_train = []
Y_train = []

# Sci-kit learn tutorial

# line_format = repr(source[0])
for line in source:
    # puts phrase id, sentence id, sentence, assignment in list
    line_elts = line.split('\t')[2:]
    # gets rid of newline in last element (assignment)
    line_elts[1] = int(line_elts[1][:-1])
    X_train.append(line_elts[0])
    Y_train.append(line_elts[1])

test_data_path = "../data/test.tsv"
source = open(test_data_path, 'r').readlines()
# get rid of title
source = source[1:]

X_test = []
Y_expected = []
Y_predicted = []

# Sci-kit learn tutorial

# line_format = repr(source[0])
for line in source:
    # puts phrase id, sentence id, sentence, assignment in list
    line_elts = line.split('\t')[2:]
    X_test.append(line_elts[0][:-1]) #-1 to get rid of newline
    #X_test.append(line)



#tokenization
# builds a dictionary of features and transforms documents to feature vectors
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train) # potentially problematic

# downscale weights for words that occur in many documents in the corpus and are therefore less informative
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)


#support vector machine
# setting up classifier
clf = svm.SVC(gamma=0.1, C=300)
#leave 10 of train for testing for now
#takes 40 min
clf.fit(X_train_tf,Y_train)
predicted = clf.predict(X_test)
svm_res = open("sentiment_analysis_results/driving_results_svm.tsv", "w")
for doc, category in zip(X_test, predicted):
    svm_res.write('%r => %s\n' % (doc, predicted[category]))
svm_res.close()
#print(test)

#test_data = []


'''
#RandomForestAlg
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train_counts, Y_train)
# testing on X_train[:-1]
test_vec = count_vect.transform(X_test)
predicted = text_classifier.predict(test_vec)
print("test",predicted)
count_res = open("sentiment_analysis_results/driving_results_randomforest.tsv", "w")
for doc, category in zip(X_test, predicted):
    count_res.write('%r => %s\n' % (doc, predicted[category]))
count_res.close()
'''

'''
#tokenization
# builds a dictionary of features and transforms documents to feature vectors
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train) # potentially problematic

# downscale weights for words that occur in many documents in the corpus and are therefore less informative
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

# fitting Naive Bayes
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
docs_new = X_test
#print("doc",docs_new)
clf = MultinomialNB().fit(X_train_tfidf, Y_train) # change #s

test_counts = count_vect.transform(docs_new)
test_tfidf = tfidf_transformer.transform(test_counts)

predicted = clf.predict(test_tfidf)
'''

'''
# submission file
new_file = open("sentiment_analysis_results/driving_results_NB.tsv", "w")

for doc, category in zip(X_test, predicted):
    new_file.write('%r => %s\n' % (doc, predicted[category])) #was Y_train
# to do: make submission file cleaner
new_file.close()
'''

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
    return X_train_counts

def train_svm():
    X_train, Y_train = import_training_data(training_data_path)
    X_test = import_test_data(test_data_path)
    X_train_counts = vectorize_sentences(X_train)

    # downscale weights for words that occur in many documents in the corpus and are therefore less informative
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)

    # setting up classifier and fitting
    clf = svm.SVC(gamma=0.1, C=300)
    clf.fit(X_train_tf,Y_train)

    # prediction
    predicted = clf.predict(X_test)

    # writing to file - return X_test, predicted. Take in file name
    svm_res = open("sentiment_analysis_results/driving_results_svm.tsv", "w")
    for doc, category in zip(X_test, predicted):
        svm_res.write('%r => %s\n' % (doc, predicted[category]))
    svm_res.close()
 

def train_nb():
    X_train, Y_train = import_training_data(training_data_path)
    X_test = import_test_data(test_data_path)
    X_train_counts = vectorize_sentences(X_train)
    pass

def train_rand_forest():
    X_train, Y_train = import_training_data(training_data_path)
    X_test = import_test_data(test_data_path)
    X_train_counts = vectorize_sentences(X_train)
    pass

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
