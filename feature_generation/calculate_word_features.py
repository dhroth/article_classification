import csv
import glob
from collections import Counter
import numpy as np
import re
import scipy.sparse
from storage import article_store
import sys
import os
import utils

#regular expression meaning "neither whitespace character nor word character"
no_specialchars_re = re.compile("[^\s\w]+")
def strip_specialchars(string):
    #return a string where all spcecial characters have been replaced with ""
    return no_specialchars_re.sub("", string)

#read in the text files in preprocessed_folder, where "read in" means:
#count how many times each word appears in each document in article_wcs
#including every word that appears in any document in vocab
#also keep track of the overall number of occurrences of each word in total_wc 
#the optional parameter valid_article_ids gives a list of text file names (without
#the .txt extension) which this function is allowed to read in. If not supplied,
#all files in the supplied folder will be read in.
def read_preprocessed_textfiles(preprocessed_folder, valid_article_ids = None):
    #article_wcs is of the form {article_id: wcs} where wcs is a Counter with words
    #as keys and counts as values
    article_wcs = {}
    #vocab contains all words which occur in any document
    vocab = set()
    #total_wc is a Counter with words as keys and overall counts as values
    total_wc = Counter()

    #read the preprocessed text from file
    text_filenames = glob.glob(preprocessed_folder + "*.txt")

    one_percent = len(text_filenames) / 100.0;
    i = 0
    for text_filename in text_filenames:
        article_id = utils.noext(text_filename)
        if valid_article_ids == None or article_id not in valid_article_ids:
            #count this one as already processed (since our caller didn't want results
            #from this file anyway)
            i += 1
            continue
        if i % int(one_percent + 1) == 0: #add 1 to avoid div by 0 error for 
                                          #one_percent < 1
            percent_done = str(int(i / one_percent) + 1)
            print "Reading in preprocessed files: ", percent_done, "% complete\r",
        sys.stdout.flush()
        with open(text_filename, "r") as text_file:
            contents = text_file.read()
            no_special_chars = strip_specialchars(contents).lower()
            words = no_special_chars.split()
            #initialize the Counter for this article_id
            article_wcs[article_id] = Counter()
            for word in words:
                vocab.add(word)
                total_wc[word] += 1
                article_wcs[article_id][word] += 1
        i += 1

    vocab = list(vocab)
    print

    return (article_wcs, vocab, total_wc)

def calculate_word_features(word_feature_specs, first_pages_folder):
    #read in the words we're supposed to use from the specs file
    specs_filename = "features/words/spec_words" + str(word_feature_specs) + ".csv"
    words = set()
    with open(specs_filename) as specs_file:
        for line in specs_file:
            words.add(line.strip())
    
    #read in the preprocessed text files
    article_wcs, vocab, total_wc = read_preprocessed_textfiles(first_pages_folder)

    #only include words if they were in the spec file
    word_feats = {}
    for article_id in article_wcs:
        word_feats[article_id] = Counter()
        for word in article_wcs[article_id]:
            if word in words:
                word_feats[article_id][word] = article_wcs[article_id][word]
    return word_feats

if __name__ == "__main__":
    #get a list of all article ids which we will calculate features for. We ignore
    #article ids which are not preprocessed or which have invalid versions
    all_article_ids = article_store.get_train_article_ids()
    article_ids_without_version = \
            set([article_id for article_id in all_article_ids 
                 if article_store.get_version(article_id) == article_store.INVALID_VERSION])
    preprocessed_article_ids = set([utils.noext(filename)
                                    for filename in glob.glob("preprocessed/first_pages/*.txt")])
    article_ids = list((all_article_ids & preprocessed_article_ids) - article_ids_without_version)
    article_id_lookup = {article_ids[i]: i for i in range(len(article_ids))}

    #read in the text files from preprocessed/first_pages/
    #supply the second argument to ignore test article ids which may have been preprocessed
    article_wcs, vocab, total_wc = \
            read_preprocessed_textfiles("preprocessed/first_pages/", set(article_ids))
    print "calculating features..."

    #only include words in the reduced_vocab if they occur a total of at least N 
    #times in any document
    word_count_cutoff = 10
    reduced_vocab = [word for word in vocab if total_wc[word] > word_count_cutoff]

    #make reduced_vocab into a lookup so we can easilly figure out what id a word is
    reduced_vocab_lookup = {reduced_vocab[i]: i for i in range(len(reduced_vocab))}


    #calculate the "separation" score for each word and only take the top N scorers

    #first, we construct a matrix X whose ith row contains the counts for the ith
    #article id (i.e. article_ids[i]) and whose jth column contains the counts
    #for the jth word (i.e. reduced_vocab[j])
    #X is sparse, so in order to save on memory use, we construct it as a scipy sparce matrix
    #in particular, we construct X as a coo_matrix because those are optimized for
    #creating matrices from individual entries, and then we convert to a csc matrix
    #because those are faster to operate on later
    i_s = []
    j_s = []
    data = []
    for article_id in article_ids:
        i = article_id_lookup[article_id]
        for word in article_wcs[article_id]:
            #throw out words not in the reduced_vocab (no point calculating separations
            #if there are only a couple documents that even have the word in it)
            if word not in reduced_vocab_lookup:
                continue
            j = reduced_vocab_lookup[word]
            i_s.append(i)
            j_s.append(j)
            data.append(article_wcs[article_id][word])
    X = scipy.sparse.coo_matrix((data,(i_s,j_s)),
                                shape=(len(article_ids),len(reduced_vocab)))
    X = X.tocsc()

    #construct a np array y whose ith entry is the version of article_ids[i]
    y = np.zeros(len(article_ids))
    for i in range(len(article_ids)):
        y[i] = article_store.get_version(article_ids[i])

    #actually calculate the separation scores
    separations = Counter()
    for j in range(len(reduced_vocab)):
        #word_frequencies is the jth column of X -- i.e. a list of how often the 
        #jth word appears in every article_id
        word_frequencies = X[:,j].toarray()
        #partition word_frequencies into two arrays -- one which only has counts for 
        #publisher versions and one which only has counts for manuscript versions 
        publisher_frequencies = word_frequencies[y == article_store.PUBLISHER_ID]
        manuscript_frequencies = word_frequencies[y == article_store.MANUSCRIPT_ID]
        #calculate the separation score. This score is large for a particular word when 
        #the distributions of counts for publisher versions and for manuscript versions 
        #are well-separated 
        separations[reduced_vocab[j]] = \
                pow(
                    (publisher_frequencies.mean() - manuscript_frequencies.mean()) /
                    (publisher_frequencies.std() + manuscript_frequencies.std()) 
                    ,2)

    #output counts for the N words with the top separation scores
    num_words_to_use = 500
    word_features_specs = str(num_words_to_use)

    most_separated_words = [tup[0] for tup in separations.most_common(num_words_to_use)]
    most_separated_words_set = set(most_separated_words)

    #write the counts for most_separated_words to the output feature file
    #use a different file name for different num_words_to_use so you can have  
    #multiple sets of features saved simultaneously
    feature_file_path = "features/words/feature_words" + word_features_specs + ".csv"
    
    #create the directory containing feature_file_path if necessary
    if not os.path.isdir(os.path.dirname(feature_file_path)):
        os.makedirs(os.path.dirname(feature_file_path))

    with open(feature_file_path, "w") as feature_file:
        writer = csv.DictWriter(feature_file, 
                                ["article_id"] + most_separated_words, 
                                lineterminator="\n")
        writer.writeheader()
        for article_id in article_ids:
            row = {"article_id":article_id}
            for word in article_wcs[article_id]:
                if word in most_separated_words_set:
                    row[word] = article_wcs[article_id][word]
            writer.writerow(row)

    spec_filename = "features/words/spec_words" + word_features_specs + ".csv"
    with open(spec_filename, "w") as spec_file:
        for word in most_separated_words_set:
            spec_file.write(word + "\n")
