import glob
import os

"""
In order for you to run any of the scripts on your own data, you must create an
interface between your data (i.e. PDFs and their metadata) and the rest of the
code in the project. Creating this interface is the job of the functions in this
file. This sample file assumes that your data is stored in a particular way
(described below). If you want this program to interface with your existing data
storage solution, you will need to modify the functions below accordingly.
Alternatively, you can export your data in such a way that it conforms to the
expectations of this script.

This script currently assumes a very simple data structure. Articles are stored
in the directory base_path (which is currently set to storage/pdfs), and each
article is a PDF with the filename <article_id>_<version>.pdf, where
article_id is a unique identifier for the article, and version is either 0 or 1
indicating (respectively) publisher or manuscript version. The article_id must
not contain any underscores in it, but otherwise can be any string which is a
valid file name.

This file works closely with storage/generate_test_set.py, which randomly
chooses a subset of articles to designate as the test set. It is important to
set aside a portion of the data as a test set before training a model so that
you can get a realistic estimate of the model's accuracy on data which was not
at all involved in the training process.
"""


PUBLISHER_ID = 0
MANUSCRIPT_ID = 1
INVALID_VERSION = 2

#this is the path (starting at the project root) to where all the PDFs are stored 
#TODO make pdf_info["path"] return the path relative to the project root instead of
#relative to base_path
base_path = "storage/pdfs/" 

#the following code runs whenever this module is imported

#first, load in the blacklist
blacklist = set()
if os.path.isfile("storage/blacklist.csv"):
    with open("storage/blacklist.csv", "r") as blacklist_file:
        for line in blacklist_file:
            blacklist.add(line.strip())

#now read through the directory base_path to look for articles
all_article_paths = glob.glob(base_path + "/*.pdf")
all_article_ids = {}
for article_path in all_article_paths:
    #the path will look like base_path/articleid_version.pdf
    #so .split("/")[-1] gives articleid_version.pdf 
    filename_components = article_path.split("/")[-1].split("_")
    article_id = filename_components[0]
    if article_id in blacklist:
        continue
    
    #cut off the .pdf extension
    version = filename_components[1][:-4]
    if version != str(PUBLISHER_ID) and version != str(MANUSCRIPT_ID):
        version = INVALID_VERSION
    else:
        version = int(version)
    if article_id in all_article_ids:
        #duplicate article id! We could throw an error, but for simplicity just
        #ignore any duplicates
        continue
    all_article_ids[article_id] = {"version": version, 
                                   "path": os.path.basename(article_path)}

#figure out which article ids are reserved for the test set
test_article_ids = set()
if os.path.isfile("storage/test_set.csv"):
    with open("storage/test_set.csv", "r") as test_set_file:
        for line in test_set_file:
            test_article_ids.add(line.strip())

###
# THE FOLLOWING THREE FUNCTION SHOULD RETURN SETS, NOT LISTS
###

#this function should return both train and test article ids
def get_all_article_ids():
    return set(all_article_ids.keys())

#this function should return only the test article ids
def get_test_article_ids():
    return test_article_ids

#this function should return only the train article ids
def get_train_article_ids():
    all_article_ids = get_all_article_ids() 
    test_article_ids = get_test_article_ids()
    return all_article_ids - test_article_ids

#this function should return a dictionary containing information about the supplied
#article id. The returned dictionary must contain at least two fields:
#"version" and "path". You may decide to add additional fields if you add functionality
#which uses them in other scripts
def get_article_id_info(article_id):
    if article_id not in all_article_ids:
        return {}
    else:
        return all_article_ids[article_id]


#this should return either PUBLISHER_ID, MANUSCRIPT_ID, or INVALID_VERSION
#we include INVALID_VERSION as a convenience, but currently anything with an
#invalid version is disregarded by the machine learning model
def get_version(article_id):
    if article_id not in all_article_ids:
        #you probably want to throw an error here, but to keep this sample file simple
        #we just return INVALID_VERSION
        return INVALID_VERSION
    else:
        return all_article_ids[article_id]["version"]
