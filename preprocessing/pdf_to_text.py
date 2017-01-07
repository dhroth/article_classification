from __future__ import division
from __future__ import print_function
import glob
from preprocessing.layout_scanner import layout_scanner
from pdfminer.pdfparser import PDFSyntaxError
from multiprocessing import Pool
from storage import article_store
import sys
import os
import traceback
import utils

#this worker function converts the first page of the pdf located at pdf_path 
#into plaintext and stores the result in plaintext_path
def parse_pdf_worker(pdf_info):
    plaintext_path = pdf_info["plaintext_path"]
    pdf_path = pdf_info["pdf_path"]
    if not os.path.isfile(pdf_path):
        #this should never happen since the main thread will only ever give 
        #valid PDFs to convert
        sys.stderr.write("Not a valid file: " + pdf_path + "\n")
        return
    try:
        first_page = layout_scanner.get_first_page(pdf_path, 
                                                   images_folder = None)
    except PDFSyntaxError as e:
        sys.stderr.write("PDFSyntaxError when parsing file " + 
                         pdf_path + "\n")
        return
    except Exception as e:
        sys.stderr.write("Unknown exception when parsing file " 
                         + pdf_path + "\n")
        sys.stderr.write(traceback.format_exc(e))
        return
    
    if not isinstance(first_page, list):
        sys.stderr.write("layout_scanner.parse_first_page returned an " + 
                         "invalid result for file " + 
                         pdf_path + "\n")
        return
    
    #in case layout_scanner.get_first_page returns an array with multiple elements 
    # TODO does this happen? Why? Do we only need the first element?
    first_page_flat = " ".join(first_page) 

    if(len(first_page) != 1):
        sys.stderr.write("first_page has length " + str(len(first_page)) + "\n")
        sys.stderr.write(str(pdf_info))
        sys.stderr.write("\n")

    with open(plaintext_path, "w") as plaintext_file:
        plaintext_file.write(first_page_flat)

def article_id_to_plaintext_path(article_id):
    return "preprocessed/first_pages/" + str(article_id) + ".txt"
 
def do_preprocessing(pdf_infos):
    # TODO should put this in utils and use the same method here as in pdf_to_img.py
    total_jobs = len(pdf_infos)
    pool = Pool(4)
    for i, _ in enumerate(pool.imap_unordered(parse_pdf_worker, pdf_infos), 1):
        print(" ", i, "/",total_jobs,"\t(",(format(i/total_jobs*100,'.2f')),"%) complete", end="\r")
        sys.stdout.flush()
    pool.close()
    pool.join()
    print()

def preprocess_text(pdf_folder, output_folder):
    pdf_paths = glob.glob(pdf_folder + "*.pdf")
    pdf_infos = [{"plaintext_path": output_folder + utils.noext(pdf_path) + ".txt",
                  "pdf_path": pdf_path}
                 for pdf_path in pdf_paths]
    do_preprocessing(pdf_infos)

if __name__ == "__main__":
    article_ids = article_store.get_train_article_ids()

    text_folder = "preprocessed/first_pages"
    if os.path.isdir(text_folder):
        #there is a (possibly full) text folder already
        #ask the user what to do
        if len(glob.glob(text_folder + "/*")) > 0:
            print ("There is already a folder containing preprocessed " + \
                   "articles. All files in the directory " + text_folder + \
                   " will be deleted. Do you wish to continue?")
            s = ""
            while s != "y" and s != "n":
                print("Continue? (y/n) ", end="")
                s = raw_input()
            if s == "y":
                for path in glob.glob(text_folder + "/*"):
                    os.unlink(path)
            else:
                print("script terminated by user")
                exit()
    else:
        #this is the first time running the preprocessing or the folder
        #was deleted, so make the proper folder
        os.makedirs(text_folder)
    
    # TODO passing args here should be the same as pdf_to_img.py
    pdf_infos = [{"plaintext_path":article_id_to_plaintext_path(article_id),
                  "pdf_path":article_store.base_path + 
                             article_store.get_article_id_info(article_id)["path"]}
                 for article_id in article_ids]
    
    do_preprocessing(pdf_infos) 
