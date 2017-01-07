from __future__ import print_function
from storage import article_store
from multiprocessing import TimeoutError
from multiprocessing import Pool
from collections import Counter
import glob
import csv
from PyPDF2 import PdfFileReader, PdfFileWriter
from PyPDF2.utils import PdfReadError
from datetime import datetime
import sys
import utils
import os
import traceback

def get_metadata(filename):
    #keep track of how long it took to parse this pdf
    start_time = datetime.now()
    #the metadata dictionary contains the metadata fields we're going to return
    metadata = {}
    #doc_info is the raw pdf DocumentInfo
    #we don't return the raw DocumentInfo because some of the fields
    #may be unparseable or have zero length
    doc_info = {}
    try:
        pdf = PdfFileReader(open(filename, "rb"))
        doc_info = pdf.getDocumentInfo()
        #there are lots of errors that can be thrown...
    except UnicodeDecodeError as e:
        sys.stderr.write("UnicodeDecodeError when parsing " + filename + "\n")
        pass
    except IOError as e:
        sys.stderr.write("IOError when parsing " + filename + "\n")
        pass
    except ValueError as e:
        sys.stderr.write("ValueError when parsing " + filename + "\n")
        pass
    except KeyError as e:
        sys.stderr.write("KeyError when parsing " + filename + "\n")
        pass
    except PdfReadError as e:
        sys.stderr.write("PdfReadError in pdf " + filename + "\n")
        pass
    except Exception as e:
        #pdfs are quite annoying and PyPDF2 has bugs, so include a catch-all
        sys.stderr.write("Unknown exception when parsing " + filename + "\n")
        sys.stderr.write(traceback.format_exc(e) + "\n")
        pass

    if doc_info is not None:
        for field in doc_info:
            try:
                #basestring is a superclass of str and unicode
                #we want to include a doc_info field if it's either a non-unicode
                #string OR if it's a unicode string that we can successfully parse
                #we don't include metadata fields with no value
                if isinstance(doc_info[field], basestring) and len(doc_info[field]) > 0:
                    metadata[field] = str(doc_info[field]) 
                else:
                    #the doc_info field is some strange type or had zero length
                    #so we don't include it in the metadata dictionary
                    pass
            except (UnicodeEncodeError, ValueError) as e:
                #ignore fields that have unparseable values
                sys.stderr.write("Could not parse a metadata field " + 
                                 field + " in file " + filename + "\n")
            except PdfReadError as e:
                #I don't know why this gets thrown here but sometimes it seems to be?
                sys.stderr.write("PdfReadError while parsing metadata field " + 
                                 field + " in file " + filename + "\n")
                sys.stderr.write(traceback.format_exc(e) + "\n")
    else:
        sys.stderr.write("Error: docinfo for " + filename + " is None" + "\n")

    elapsed_time = (datetime.now() - start_time)
    #uncomment these lines if you want to watch as every pdf gets parsed
    #print("parsed pdf " + filename + " in " + \
    #    str(elapsed_time.seconds) + "." + str(elapsed_time.microseconds) + " seconds. " + \
    #    "Found " + str(len(metadata.keys())) + " metadata fields")
        
    #if there was an error while extracting the doc_info or while parsing through
    #the doc_info fields, we will return an empty dictionary
    if metadata == {}:
        sys.stderr.write(filename + " returned an empty metadata dict\n")
    return metadata


#make a metadata field name into something that can be made into a filename
def safify_field(fieldname):
    keepcharacters = (' ','.','_')
    safe_field = "".join([c for c in fieldname if c.isalnum() or c in keepcharacters]) \
                   .rstrip()
    return safe_field
 

def calculate_metadatas(paths):
    #the task of reading and parsing PDFs is (at least on my disk) almost completely
    #disk-bound. I found that I got the biggest throughput on my machine when running
    #only one process at a time. If you are working with an SSD, a disk with multiple
    #read heads, or files stored across multiple disks, you may benefit from using
    #more worker processes
    pool = Pool(1)
    results = []
    it = pool.imap(get_metadata, paths)
    one_percent = len(paths) / 100.0
    for i in range(len(paths)):
        #do not set a timeout here! The iterator does not advance if the timeout
        #is triggered and we rely on the results being in the same order as the
        #paths which were passed in
        result = it.next()
        print(str(int(i / one_percent)) + "% complete", end="\r")
        results.append(result)
    pool.close()
    pool.join()
    print()
    
    return results


#returns (fields, words, metadatas) where:
# fields is a set of all metadata field names for the supplied specs
# words is a dict mapping field names to words in the field
# metadatas is a dict: metadatas[path_id][field_name][word]
#  which specified the number of times a word appears in field_name for pdf_id
def calculate_metadata_features(specs, pdf_folder):
    spec_filename = "features/metadata/specs_" + specs + ".csv"
    #num_top_words = specs.split("_")[0][:-1]
    #field_min_percent_occurrence = specs.split("_")[1][:-1]
   
    fields = set()
    words = {}
    with open(spec_filename, "r") as spec_file:
        spec_reader = csv.DictReader(spec_file)
        for line in spec_reader:
            field = line["field"].strip()
            word = line["word"].strip()
            fields.add(field)
            if field not in words:
                words[field] = set()
            words[field].add(word)

    paths = glob.glob(pdf_folder + "*.pdf")
    results = calculate_metadatas(paths)
     
    metadatas = {}
    for i in range(len(results)):
        path_id = utils.noext(paths[i])
        metadatas[path_id] = {}
        for field in results[i]:
            safe_field = safify_field(field)
            if safe_field in fields:
                metadatas[path_id][safe_field] = Counter()
                for word in results[i][field].split(" "):
                    if word in words[safe_field]:
                        metadatas[path_id][safe_field][word] += 1
    return (fields, words, metadatas)

if __name__ == "__main__":
    article_ids = list(article_store.get_train_article_ids())
    
    #for each metadata field F, we create a feature for each article which is 
    #the number of times word W appears in the value of field F. We only include 
    #features for the top num_top_words words 
    num_top_words = 100

    #only include metadata fields in our final output which appear in at least
    #field_min_percent_occurrence percent of all the PDFs we parse
    field_min_percent_occurrence = 0.01

    #store the feature files in the directory feature_folder, which we name after
    #the number of top words and the minimum cutoff prevalence for fields we used
    metadata_spec = str(num_top_words) + "w_" + \
                    str(field_min_percent_occurrence) + "f"
    feature_folder = "features/metadata/" + metadata_spec
    
    if os.path.isdir(feature_folder):
        #this feature folder already exists, so ask the user if we should overwrite
        #everything. If we don't overwrite everything, we may end up with an incomplete 
        #set of feature files from an older data set mixed in with the feature files 
        #that we're about to generate basesd on the current data
        print ("A feature folder for the parameters specified in the " + \
              "script already exists. In order to proceed, the csv files " + \
              "in this directory must be erased. Consider renaming the " + \
              "existing directory if you wish to continue without erasing. ")
        s = ""
        while s != "y" and s != "n":
            print("Continue? (y/n) ", end="")
            s = raw_input()
        if s == "y":
            #delete all the csv files in the feature folder
            for path in glob.glob(feature_folder + "/*.csv"):
                os.unlink(path)
        else:
            print("script terminated by user")
            exit()
    else:
        #a feature folder doesn't already exist for these parameters, so make one
        os.makedirs(feature_folder)
     
    #get the pdf path for each article to pass to the worker function get_metadata()
    article_id_infos = [article_store.get_article_id_info(article_id)
                        for article_id in article_ids]
    paths = [article_store.base_path + article_id_infos[i]["path"] 
             for i in range(len(article_ids))]

    #calculate metadatas
    results = calculate_metadatas(paths)

    metadatas = {article_ids[i]: results[i] for i in range(len(article_ids))}

    #for each metadata field, calculate how many PDFs contained that field and
    #store the result in field_num_occurrences[field]
    field_num_occurrences = Counter()
    for article_id in metadatas:
        fields = metadatas[article_id].keys()
        for field in fields:
            field_num_occurrences[field] += 1

    #decide which fields occur often enough in the PDF metadata to be 
    #included in the feature set. Store fields which occur in at least 
    #field_min_percent_occurrence percent of all the PDFs we parsed in the
    #set top_fields
    top_fields = set([])
    for field in field_num_occurrences:
        # XXX
        if field_num_occurrences[field] > \
           len(metadatas.keys()) * field_min_percent_occurrence:
            top_fields.add(field)

    #store the total number of times that word W appeared in field F across all
    #articles in word_counts[F][W]
    word_counts = {}
    for field in top_fields:
        word_counts[field] = Counter()

    for article_id in metadatas:
        for field in top_fields:
            if field in metadatas[article_id]:
                words = metadatas[article_id][field].split(" ")
                for word in words:
                    word_counts[field][word] += 1

    #store the top num_top_words most common words which occur in field F across
    #all articles in top_words[F]
    top_words = {}
    for field in word_counts:
        top_words[field] = [tup[0] for tup in 
                            word_counts[field].most_common(num_top_words)]
    
    #for each field, write a feature file
    #include columns for each top word along with a not_present col
    #to indicate the pdf didn't have any of the top words for that metadata field
    for field in top_fields:
        cols = ["article_id"] + top_words[field] + ["not_present"]

        #create a safe filename based on the metadata field name
        safe_field = safify_field(field) 
        feature_filename = feature_folder + "/feature_" + safe_field + ".csv"
        
        with open(feature_filename, "w") as feature_file:
            writer = csv.DictWriter(feature_file, cols, lineterminator="\n")
            writer.writeheader()
            for article_id in metadatas:
                #initialize row to all 0s
                row = {}
                for col in cols:
                    row[col] = 0

                #write the article id and then the count for each word
                row["article_id"] = article_id
                    
                if field in metadatas[article_id]:
                    words = metadatas[article_id][field].split(" ")
                    for word in words:
                        if word in top_words[field]:
                            row[word] += 1
                else:
                    row["not_present"] = 1
                writer.writerow(row)
                
    #write a metadata field spec file so we can calculate the exact same features
    #on more PDFs even after this script exits
    #each row of spec_filename has a field and a top word corresponding to that field
    spec_filename = "features/metadata/specs_" + metadata_spec + ".csv"
    with open(spec_filename, "w") as spec_file:
        cols = ["field", "word"]
        spec_writer = csv.DictWriter(spec_file, cols, lineterminator="\n")
        spec_writer.writeheader()
        for field in top_fields:
            safe_field = safify_field(field)
            for word in top_words[field]:
                spec_writer.writerow({"field": safe_field, "word": word})
