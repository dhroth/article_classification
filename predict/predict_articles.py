import glob
import sys
import os
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from pprint import pprint
import utils
import csv
from scipy.sparse import coo_matrix

from preprocessing.pdf_to_text import preprocess_text
from preprocessing.pdf_to_img import preprocess_img

from feature_generation.calculate_metadata_features import calculate_metadata_features
from feature_generation.calculate_word_features import calculate_word_features
from feature_generation.calculate_img_features import calculate_img_features

from feature_generation.calculate_metadata_features import safify_field

metadata_feature_specs = "100w_0.01f"
word_feature_specs = 500
img_feature_specs = "default"

model_name = metadata_feature_specs +  "__" + \
             str(word_feature_specs) + "__" + \
             img_feature_specs

#the dpi to preprocess our predictees at
dpi = 20

#do preprocessing for everything in predictee_folder
predictee_folder = #path to directory with PDFs to predict

output_file = #file where the results should be written

#load the pickled model for this particular set of features
model_folder = "trained_models/" + model_name + "/"
try:
    feature_names_lookup = joblib.load(model_folder + "features.pkl")
    clf = joblib.load(model_folder + "model.pkl")
except IOError:
    exit("Could not find pkl files for the model parameters specified")
except Exception:
    exit("Could not read in pkl files. They may have been corrupted")

#creates a temporary directory if it doesn't exist and cleans it out if it does 
def prep_tmpdir(dirname):
    if os.path.isdir(dirname):
        for path in glob.glob(dirname + "/*"):
            os.unlink(path)
    else:
        os.mkdir(dirname)

prep_tmpdir("predict/tmp_first_pages")
prep_tmpdir("predict/tmp_images")

#return the filename sans extension
def get_predictee_id(predictee):
    return os.path.basename(predictee)[:-4]

print "extracting text from PDFs..."
preprocess_text(predictee_folder, "predict/tmp_first_pages/")
print "generating " + str(dpi) + " dpi images from PDFs..."
preprocess_img(dpi, predictee_folder, "predict/tmp_images/")

print "calculating metadata features..."
metadata_fields, metadata_words, metadata_feats = \
        calculate_metadata_features(metadata_feature_specs, predictee_folder)
print "calculating word features..."
word_feats = calculate_word_features(word_feature_specs, "predict/tmp_first_pages/")
print "calculating img features..."
img_feats = calculate_img_features(img_feature_specs, "predict/tmp_images/")

#now construct an X matrix that we can run through clf
pdf_paths = glob.glob(predictee_folder + "*.pdf")

#feature names in feature_names_lookup are prepended with the filename they came from
#our features didn't come from a file but we know what the file names would be if they did:
metadata_features_filenames = {fieldname: "features/metadata/" + metadata_feature_specs + \
                                          "/feature_" + safify_field(fieldname) + ".csv"
                               for fieldname in metadata_fields}
word_features_filename = "features/words/feature_words" + str(word_feature_specs) + ".csv"
img_features_filename = "features/img/feature_img_" + img_feature_specs + ".csv"

i_s = []
j_s = []
data = []
#populate the feature matrix X
#default feature values to 0 if no features were calculated
for i in range(len(pdf_paths)):
    pdf_path = pdf_paths[i]
    pdf_id = utils.noext(pdf_path)
    #populate the feature matrix with metadata features
    if pdf_id in metadata_feats:
        for field in metadata_fields:
            feature_filename = metadata_features_filenames[field]
            if field in metadata_feats[pdf_id]:
                #the field does appear in the metadata for this PDF
                #so make features for each word in the field
                for word in metadata_feats[pdf_id][field]:
                    feature_name = feature_filename + "_" + word
                    j = feature_names_lookup[feature_name]
                    feature_value = metadata_feats[pdf_id][field][word]
                    if feature_value != 0:
                        i_s.append(i)
                        j_s.append(j)
                        data.append(feature_value)
            else:
                #the field was not in the PDF's metadata key value store
                #so include the not_present feature with value 1
                i_s.append(i)
                j = feature_names_lookup[feature_filename + "_not_present"]
                j_s.append(j)
                data.append(1)
    #populate the feature matrix with word features
    if pdf_id in word_feats:
        for word in word_feats[pdf_id]:
            feature_name = word_features_filename + "_" + word
            j = feature_names_lookup[feature_name]
            feature_value = word_feats[pdf_id][word]
            if feature_value != 0:
                i_s.append(i)
                j_s.append(j)
                data.append(feature_value)
    #populate the feature matrix with img features
    if pdf_id in img_feats:
        for img_feature in img_feats[pdf_id]:
            feature_name = img_features_filename + "_" + img_feature
            j = feature_names_lookup[feature_name]
            feature_value = img_feats[pdf_id][img_feature]
            if feature_value != 0:
                i_s.append(i)
                j_s.append(j)
                data.append(feature_value)

X_shape = (len(pdf_paths), len(feature_names_lookup))
X = coo_matrix((data,(i_s,j_s)), shape = X_shape)
X = X.tocsr()

#do the actual prediction
predictions = clf.predict_proba(X)

#write the predictions to predict/predictions.csv
with open(output_file, "w") as predictions_file:
    predictions_writer = csv.DictWriter(predictions_file,
                                        fieldnames = ["pdf_name", "prediction"],
                                        lineterminator = "\n")
    predictions_writer.writeheader()

    for i in range(len(pdf_paths)):
        row = {"pdf_name": utils.noext(pdf_paths[i]),
               "prediction": predictions[i][1]}
        predictions_writer.writerow(row)

