from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import cross_validation
import random
import csv
import glob
import math
import numpy as np
import utils
from storage import article_store
from datetime import datetime
from scipy.sparse import coo_matrix
import sys
from pprint import pprint
import os

log = utils.log

log("start")

PUBLISHER_ID = article_store.PUBLISHER_ID
MANUSCRIPT_ID = article_store.MANUSCRIPT_ID
INVALID_VERSION = article_store.INVALID_VERSION

###### READ IN/SET CONFIGURATION PARAMETERS ######

#the predicted class probability must be > this in order to make a prediction
if len(sys.argv) != 2:
    exit("ERROR: format is python -m model <required_confidence>")
required_confidence = float(sys.argv[1]) 

metadata_features_specs = "100w_0.01f"
word_features_specs = 500
img_features_specs = "default"

#number of folds to use for cross validation
xval = 5

#a flag to indicate whether or not we want to save the model trained in this run 
save_model = True

#flag to indicate whether or not to write the model's performance to the
#performance file (default results/performance.csv)
write_performance = True

model_name = metadata_features_specs + "__" + \
             str(word_features_specs) + "__" + \
             img_features_specs


#human-readable description for performance file
features_description = "metadata type: " + metadata_features_specs + \
                       ", #words: " + str(word_features_specs) + \
                       ", img type: " + img_features_specs


if save_model:
    if os.path.isdir("trained_models/" + model_name):
        #we have to delete all contents of this directory lest the files which are already
        #there conflict with the files we write later, which would prevent the model.pkl
        #from being read properly later on
        print "A model trained with these parameters may already exist. Continuing will " + \
              "overwrite any such trained model (located in the trained_models/ directory)."
        s = ""
        while s != "y" and s != "n":
            print "Continue? (y/n) ",
            s = raw_input()
        if s == "y":
            for path in glob.glob("trained_models/" + model_name + "/*"):
                os.unlink(path)
        else:
            exit("script terminated by user")
    else:
        os.makedirs("trained_models/" + model_name)

######  READ IN FEATURES ######

#get a list of all feature filenames (metadata, img, and word)
metadata_features_dir = "features/metadata/" + metadata_features_specs + "/"
metadata_features_filenames = glob.glob(metadata_features_dir + "feature_*.csv")
word_features_filename = "features/words/feature_words" + str(word_features_specs) + ".csv"
img_features_filename = "features/img/feature_img_" + img_features_specs + ".csv" 

all_features_filenames = metadata_features_filenames + \
                         [img_features_filename] + [word_features_filename]

all_article_ids = article_store.get_train_article_ids()

article_ids_without_version = set([article_id for article_id in all_article_ids 
                                   if article_store.get_version(article_id) == INVALID_VERSION])

#feature_names_lookup is {feature_name:0, feature_name:1, feature_name:2, ...}
#the number associated with each feature_name is its column number in the data matrix X
feature_names_lookup = {}
#article_ids_with_features_lookup is {article_id:0, article_id:1, article_id:2, ...}
#the number associated with each article id is its row number in the data matrix X
article_ids_with_features_lookup = {}

log("loading features from feature files...")
#we construct the data matrix as a sparce coo_matrix in order to save on memory
#for each data point, we store its row (article_id) in i, its column (feature id) in j, and
#the value of the data point in data. The value of X[i[n],j[n]] is data[n]
i_s = []
j_s = []
data = []
for feature_filename in all_features_filenames:
    with open(feature_filename, "r") as feature_file:
        reader = csv.DictReader(feature_file)
        #construct a unique identifier for this feature
        feature_names = [feature_filename + "_" + fieldname for fieldname in reader.fieldnames
                         if fieldname != "article_id"]

        #add the new features to feature_names_lookup
        for feature_name in feature_names:
            if feature_name in feature_names_lookup:
                sys.stderr.write("found duplicate feature " + feature_name + ". Some " + \
                                 "feature data may be read incorrectly. Please " + \
                                 "ensure that all features have unique names " + \
                                 "within their feature file. \n")
            else:
                #we haven't seen this feature name before, so assign it a column id
                feature_names_lookup[feature_name] = len(feature_names_lookup)
       
        for line in reader:
            #there is one column for the article_id and the rest are features 
            article_id = line["article_id"]

            if article_id in article_ids_without_version:
                #ignore article_ids with no version
                continue 
            
            #if this is the first time we've seen any features for this article_id,
            #assign it a row id
            if article_id not in article_ids_with_features_lookup:
                new_i = len(article_ids_with_features_lookup) 
                article_ids_with_features_lookup[article_id] = new_i
                        

            i = article_ids_with_features_lookup[article_id]
            #now record the values of the features for this article_id
            for col_name in line:
                if col_name == "article_id":
                    #every column besides the article_id column is a feature column
                    continue
                #construct the same unique identifier for this feature as above
                feature_name = feature_filename + "_" + col_name

                j = feature_names_lookup[feature_name]
                feature_value = line[col_name]
                if feature_value == "":
                    feature_value = 0.0
                else:
                    feature_value = float(feature_value)

                if feature_value != 0:
                    #only append new data if the data is not 0 (since it's a sparse matrix)
                    i_s.append(i)
                    j_s.append(j)
                    data.append(feature_value)

log("done loading features from feature files.")
log("preparing X, y...")

#get a list of article_ids sorted by their row number
article_ids = sorted(article_ids_with_features_lookup, key=article_ids_with_features_lookup.get)

#create the data matrix X from i_s, j_s, and data
X_shape = (len(article_ids), len(feature_names_lookup))
X = coo_matrix((data,(i_s,j_s)), shape=X_shape)

#create an array with the versions of every article
y = np.zeros(len(article_ids))
for i in range(len(article_ids)):
    y[i] = article_store.get_version(article_ids[i])

log("done preparing X, y")
log("y.shape=" + str(y.shape))
log("X.shape=" + str(X.shape))

#change X to csr format for more efficient use with machine learning algorithms
X = X.tocsr()

#set class_weight to balanced to effectively equalize class sizes so learning 
#isn't biased by how many samples you have of publisher and manuscript versions
#if you choose to use a classifier that doesn't support the 
#class_weight = balanced parameter, make sure to uncomment the code below that 
#manually equalizes the class sizes

#we used logistic regression for our data, but you might try using one of the
#commented-out classifiers below or some other classifier
clf = LogisticRegression(class_weight = "balanced")
#clf = SVC(class_weight = "balanced", probability = True)
#clf = DecisionTreeClassifier(class_weight = "balanced")
#clf = RandomForestClassifier(class_weight = "balanced")


#this function (normalized_cross_val()) trains and validates the model using 
#cross-validation. For a given value of cv, this function partitions the data
#into cv distinct sets and then trains the model with cv-1 of those partitions
#it uses the last partition to calculate the accuracy of the trained model
#the fxn returns counts of false/true positives/negatives for each of the cv runs

#clf is the classifier to use
#X is the data matrix in whatever format clf accepts it in. Note that csc format
#  is undesirable because we split into train/validation sets by row
#y is an array of the correct labels
#cv is the number of cross-validations to do
def normalized_cross_val(clf, X, y, cv):
    if int(cv) != cv:
        print "invalid cv"
        return None

    if X.shape[0] != y.shape[0]:
        print "shapes don't match"
        return None
    
    #split the data into cv random partitions called folds
    num_samples = y.shape[0]
    random_indices = np.arange(num_samples)
    random.shuffle(random_indices)
    fold_indices = np.array_split(random_indices, cv)

    #for performance analysis purposes, keep track of false/true positives/negatives
    #where classifying an article as a manuscript is "positive", publisher is "negative"
    #a false positive is incorrectly classifying an article as a manuscript, etc.
    POSITIVE_ID = MANUSCRIPT_ID
    NEGATIVE_ID = PUBLISHER_ID

    false_positives = []
    false_negatives = []
    true_positives = []
    true_negatives = []
    total_positives = []
    total_negatives = []
    
    for i in range(cv):
        #use one fold for testing and the rest of the folds for training
        test_indices = fold_indices[i]
        train_indices = np.hstack(fold_indices[:i] + fold_indices[i+1:])

        manuscript_indices = [index for index in train_indices if y[index] == MANUSCRIPT_ID]
        publisher_indices = [index for index in train_indices if y[index] == PUBLISHER_ID]

        """
        #this piece of code balances the class weights.
        #we don't need it if using a scikit classifier with class_weight = balanced
        
        def expand_indices(indices, desired_length):
            num_repetitions = int(float(desired_length) / len(indices))
            num_leftover = desired_length - (len(indices) * num_repetitions)
            repetitions = np.ones(len(indices)) * num_repetitions
            leftover = np.zeros(len(indices))
            leftover[:num_leftover] = 1
            expanded = np.repeat(indices, map(int, repetitions + leftover))
            return expanded

        #duplicate either the publisher rows or the manuscript rows (depending on which 
        #is bigger) so that the sizes of both classes are the same
        if len(manuscript_indices) == len(publisher_indices):
            #neither list needs to be expanded
            pass
        else:
            if len(manuscript_indices) > len(publisher_indices):
                publisher_indices = expand_indices(publisher_indices, len(manuscript_indices))
            else:
                manuscript_indices = expand_indices(manuscript_indices, len(publisher_indices))

        train_indices = np.concatenate((publisher_indices, manuscript_indices))
        """

        #generate X and y and do the training
        X_test, y_test = X[test_indices], y[test_indices]
        X_train, y_train = X[train_indices], y[train_indices]
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)
        
        class_order = list(clf.classes_)

        positive_probas = y_proba[:,class_order.index(POSITIVE_ID)]
        negative_probas = y_proba[:,class_order.index(NEGATIVE_ID)]
        
        #only make predictions for probabilities above the cutoff confidence
        positive_predicted = positive_probas > required_confidence
        negative_predicted = negative_probas > required_confidence
        
        positive_actual = y_test == POSITIVE_ID
        negative_actual = y_test == NEGATIVE_ID
        
        #calculate false/true positives/negatives
        false_positive = (positive_predicted * negative_actual).sum()
        false_negative = (negative_predicted * positive_actual).sum()
        true_positive = (positive_predicted * positive_actual).sum()
        true_negative = (negative_predicted * negative_actual).sum()

        total_positive = positive_actual.sum()
        total_negative = negative_actual.sum()
        
        false_positives.append(false_positive)
        false_negatives.append(false_negative)
        true_positives.append(true_positive)
        true_negatives.append(true_negative)
        total_positives.append(total_positive)
        total_negatives.append(total_negative)
        
    return (np.array(false_positives),
            np.array(false_negatives),
            np.array(true_positives),
            np.array(true_negatives),
            np.array(total_positives),
            np.array(total_negatives))
    
log("running cross-validation...")
false_positives, false_negatives, \
        true_positives, true_negatives, \
        total_positives, total_negatives = normalized_cross_val(clf, X, y, cv = xval)
log("done with cross-validation")

if write_performance:
    with open("results/performance.csv", "a") as performance_file:
        performance_file.write(str(datetime.now()) + "," +
                               "Logistic Regression," +    #algorithm used
                               str(xval) + "," +   #number of folds
                               "yes," +    #classes equalized?
                               "\"" + features_description + "\"," + 
                               str(required_confidence) + ",")
        performance_file.write("\n,,,,,,".join([ \
                               str(false_positives[i]) + "," + \
                               str(false_negatives[i]) + "," + \
                               str(true_positives[i]) + "," + \
                               str(true_negatives[i]) + "," + \
                               str(total_positives[i]) + "," + \
                               str(total_negatives[i]) \
                               for i in range(len(false_positives))]) + "\n")

else:
    #write out the results to the terminal instead of to the performance file

    #figure out how often the model refused to make a prediction
    percents_no_prediction = 1.0 - \
        (   (false_positives + false_negatives + true_positives + true_negatives).astype(float) / \
            (total_positives + total_negatives)     )
    manuscript_scores = true_positives.astype(float) / (true_positives + false_negatives)
    publisher_scores = true_negatives.astype(float) / (true_negatives + false_positives)
    overall_scores = (manuscript_scores + publisher_scores) / 2.0

    print "percent no prediction:", percents_no_prediction
    print "avg:", percents_no_prediction.mean()
    print "std:", percents_no_prediction.std()

    print

    print "publisher scores on test data:", publisher_scores
    print "avg:", publisher_scores.mean()
    print "std:", publisher_scores.std()

    print

    print "manuscript scores on test data:", manuscript_scores
    print "avg:", manuscript_scores.mean()
    print "std:", manuscript_scores.std()

    print

    print "overall scores on test data:", overall_scores
    print "avg:", overall_scores.mean()
    print "std:", overall_scores.std()

if save_model:
    log("saving classifier to model.pkl...")
    from sklearn.externals import joblib
    joblib.dump(clf, "trained_models/" + model_name + "/model.pkl")
    joblib.dump(feature_names_lookup, "trained_models/" + model_name + "/features.pkl")
log("done")
