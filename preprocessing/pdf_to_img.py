from __future__ import division
from __future__ import print_function
from multiprocessing import Pool
import os
import glob
from storage import article_store
import sys
import traceback
import utils

#this worker function takes in an article_id and a pdf path and 
#converts the first page of the pdf at that path to an image
def convert(args):
    #img_path does not contain the extension since pdftoppm adds that 
    img_path = args[0] 
    #the path of the pdf that we need to turn into an image
    pdf_path = args[1] 
    dpi = args[2] 
    
    try:
        args = [
            "-f 0",
            "-singlefile",
            "-tiff",
            "-rx " + str(dpi),
            "-ry " + str(dpi),
            pdf_path,
            img_path]
        os.system("pdftoppm " + " ".join(args)) 
    except Exception as e:
        sys.stderr.write("unknown exception on img" + img_path + "; " + pdf_path)
        sys.stderr.write(traceback.format_exc(e))

def do_preprocessing(pdf_infos):
    total_jobs = len(pdf_infos)
    pool = Pool(8)
    #print out a progress bar while convert runs
    for i, _ in enumerate(pool.imap_unordered(convert, pdf_infos), 1):
        print(i,"/",total_jobs,"\t(",(format(i/total_jobs*100,'.2f')),"%) complete", end="\r")
        sys.stdout.flush()
    pool.close()
    pool.join()
    print()


def preprocess_img(dpi, pdf_folder, img_folder):
    pdf_paths = glob.glob(pdf_folder + "*.pdf")
    pdf_infos = [[img_folder + utils.noext(pdf_path), pdf_path, dpi]
                 for pdf_path in pdf_paths] 
    do_preprocessing(pdf_infos)

if __name__ == "__main__":
    #create 20dpi images
    dpi = 20 
    #if you choose to use a different dpi, the script will store
    #the images in a different folder
    images_folder = "preprocessed/images" + str(dpi)
    
    if os.path.isdir(images_folder):
        #there already exists a folder for the specified dpi
        #ask the user what to do
        print ("An image folder for the specified dpi (" + str(dpi) + \
               ") already exists. If you choose to proceed this script " + \
               "will delete all data in this directory. Consider renaming " + \
               "the existing directory if you wish to continue without erasing.")
        s = ""
        while s != "y" and s != "n":
            print("Continue? (y/n) ", end="")
            s = raw_input()
        if s == "y":
            for path in glob.glob(images_folder + "/*.tif"):
                os.unlink(path)
        else:
            print("script terminated by user")
            exit()
    else:
        #no images folder for the specified dpi exists yet, so make one
        os.makedirs(images_folder)

    #img_paths[i] and pdf_paths[i] are the path to the image that needs
    #to be created and the path to the pdf that needs to be read which 
    #correspond to article_ids[i]
    article_ids = list(article_store.get_train_article_ids())
    #img_paths does not have the .tif extension
    img_paths = [images_folder + "/" + article_id for article_id in article_ids] 
    pdf_paths = [article_store.base_path + 
                 article_store.get_article_id_info(article_id)["path"] 
                 for article_id in article_ids]
       
    #the convert function needs three arguments: the path it needs to 
    #create the image at, the path of the pdf it needs to convert, and 
    #the dpi it needs to use
    pdf_infos = [[img_paths[i], pdf_paths[i], dpi] for i in range(len(article_ids))]
    
    do_preprocessing(pdf_infos)
