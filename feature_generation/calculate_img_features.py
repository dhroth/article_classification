from __future__ import print_function
import csv
from multiprocessing import Pool
import numpy as np
import PIL.Image
import glob
from itertools import groupby
import gc
import sys
import os
import utils

def rgb_to_gray(rgb):
    #rgb is a tuple
    #we could do something fancier than the average like weighting each channel
    #by how well the eye perceives the color, but that's probably overkill

    #we don't use np.mean because it's way way slower, 
    #and I don't want to sum() in case it's rgba
    return (rgb[0] + rgb[1] + rgb[2])/3.0 

#this worker function takes an image path and returns a dictionary of features
#describing the image at that path
def calculate_features(args):
    specs = args["specs"]
    image_path = args["image_path"]
    features = {}
    try:
        with PIL.Image.open(image_path) as img:
            pixels = list(img.getdata())
            num_pixels = float(len(pixels))

            #img.size is width x height
            width = float(img.size[0])
            height = float(img.size[1])

        #sometimes when I run this script, Python seems reluctant to reuse the
        #memory which was consumed by opening up the image file. When that happened,
        #calling gc.collect() seemed to fix the issue. However, gc.collect() is
        #an expensive call, so don't uncomment this unless this script is using
        #too much memory
        #gc.collect() 
  
    except IOError as e:
        print("IOError on image" + image_path)           
        return features
 
    #to add your own features under a specs other than default, use
    if(specs == "myspec"):
        features["myfeature"] = 7 #or perhaps some other number

    #calculate average and median grayscale pixel value
    #if each pixel is a tuple then the image is rgb or rgba
    #if each pixel is a scalar, then it's already grayscale so we don't
    #need to convert
    gray_pixels = [rgb_to_gray(p) if type(p) is tuple else p for p in pixels]
    features["avg-grayscale"] = np.mean(gray_pixels)
    features["median-grayscale"] = np.median(gray_pixels)
    
    #calculate cutoff features:
    #one feature is calculated for each element in cutoffs, where the value
    #of the feature for each article is the number of pixels in the article
    #whose grayscale value is greater than the value of cutoff[i]
    cutoffs = [5,100,200,250,254]
    for cutoff in cutoffs:
        num_above_cutoff = len([p for p in gray_pixels if p > cutoff])
        features["cutoff_" + str(cutoff)] = num_above_cutoff / num_pixels

    #calculate a feature for the percent of the pixels in the image which are
    #not pure gray. If the first pixel is a tuple, assume the whole image is
    #rgb or rgba, otherwise set the percent-color to zero
    if type(pixels[0]) is tuple:
        num_color = len([p for p in pixels if p[0] != p[1] or p[1] != p[2]])
        features["percent-color"] = num_color / num_pixels
    else:
        features["percent-color"] = 0

    #now calculate features about the 2D layout of the image
    #reshape the pixels into a properly-sized 2D array
    img_array = np.array(gray_pixels).reshape(height,width)
    #turn the grayscale image into a black/white image where every pixel
    #above threshold_cutoff is considered white and every pixel below, black
    threshold_cutoff = 200
    thresholded = np.ones(img_array.shape) * (img_array > threshold_cutoff)

    #helper function which returns the length of the longest streak of 0s in 
    #the supplied 1D array
    #this function could probably be made faster, but this way is convenient
    def longest_streak(arr):
        longest = 0
        for p_val, group in groupby(arr):
            if p_val == 0:
                longest = max(longest, len(list(group)))
        return longest
    
    #calculate the longest horizontal row of black pixels in any row
    longest_hrs = []
    for row in thresholded:
        longest_hrs.append(longest_streak(row))
    longest_hr = max(longest_hrs)
    longest_hr_percent = longest_hr / width
    #save the longest streak as a percent of the total width
    features["longest-hr-percent"] = longest_hr_percent
    
    #calculate the longest vertical column of black pixels in any column
    longest_vrs = []
    for col in thresholded.T:
        longest_vrs.append(longest_streak(col))
    longest_vr = max(longest_vrs)
    longest_vr_percent = longest_vr / height
    #save the longest streak as a percent of the total height
    features["longest-vr-percent"] = longest_vr_percent


    #check for whether the document appears to be split into columns. 
    #do this by looking for consecutive columns of all white. First 
    #look in the middle of the image, where we would expect to see such 
    #white columns if there were two columns

    #look at the 10% of pixel columns centered in the image
    percent_middle = 0.1 
    center_col = width / 2
    start_col = center_col - percent_middle * 0.5 * width
    end_col = center_col + percent_middle * 0.5 * width
    middle_cols = thresholded.T[start_col:end_col]
    longest_middle_white_vrs = []
    for col in middle_cols:
        #look for streaks of white instead of black by doing 1-col
        longest_middle_white_vrs.append(longest_streak(1 - col))
    longest_middle_white_vr = max(longest_middle_white_vrs)
    longest_middle_white_vr_percent = longest_middle_white_vr / height
    features["longest-middle-white-vr-percent"] = longest_middle_white_vr_percent
   

    return features
        

def calculate_img_features(specs, img_folder):
    paths = glob.glob(img_folder + "*.tif")
    results = do_feature_calculation(specs, paths)
    img_feats = {utils.noext(paths[i]): results[i] for i in range(len(paths))}
    return img_feats

def do_feature_calculation(specs, paths):
    results = []
    pool = Pool(8)
    args = [{"specs": specs, "image_path": image_path} for image_path in paths]
    results_it = pool.imap(calculate_features, args)
    one_percent = len(paths) / 100.0
    for i, result in enumerate(results_it):
        print(str(int(i / one_percent)) + "% complete", end="\r")
        sys.stdout.flush()
        results.append(result)
    pool.close()
    pool.join()
    print()
    return results
    

if __name__ == "__main__":
    from storage import article_store 
    
    #use images of this dpi to calculate the image features
    #you must have already run preprocessing/pdf_to_img.py with the same dpi
    #or else this script will have no images to calculate features from
    dpi = 20
    img_features_specs = "default"

    article_ids = article_store.get_train_article_ids()

    article_ids_with_version = \
        [article_id for article_id in article_ids
         if article_store.get_version(article_id) != article_store.INVALID_VERSION]
    
    #img_path is the directory containing all the images with resolution dpi
    #this is where preprocessing/pdf_to_img.py will store images when you run
    #it with the same dpi
    img_path = "preprocessed/images" + str(dpi) + "/"
    all_img_paths = glob.glob(img_path + "*.tif")

    #discard any article_ids which don't have corresponding images in img_path
    #i.e. which haven't been preprocessed successfully
    article_ids_with_img = [article_id for article_id in article_ids_with_version 
                            if img_path + article_id + ".tif" in all_img_paths]
    
    image_paths = [img_path + article_id + ".tif" for article_id in article_ids_with_img]
    
    #do the actual feature calculation
    results = do_feature_calculation(img_features_specs, image_paths)

    #allow for the possibility that calculate_features() will return different 
    #features for different images by taking the union of the feature names
    #from each result in results
    all_features = set()
    for features in results:
        for feature in features:
            all_features.add(feature)

    #write the calculated features to file as a csv

    #change this file path if you want to try calculating new image features and
    #don't want to overwrite the existing img features
    feature_file_path = "features/img/feature_img_" + img_features_specs + ".csv"
    #make the directory containing feature_file_path if necessary
    if not os.path.isdir(os.path.dirname(feature_file_path)):
        os.makedirs(os.path.dirname(feature_file_path))

    with open(feature_file_path, "w") as feature_file:
        writer = csv.DictWriter(feature_file, 
                                fieldnames = ["article_id"] + list(all_features), 
                                lineterminator = "\n")
        writer.writeheader()
        for i in range(len(results)):
            features = results[i]
            features["article_id"] = article_ids_with_img[i] 
            writer.writerow(features)
