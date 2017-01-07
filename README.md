#Overall Organization
The overall purpose of this program is to take many scholarly article PDFs whose versions are known and to learn from them about how to classify previously-unseen scholarly articles. Because one is often interested in learning from a large number of PDFs, we have divided the learning process into separate stages so as to allow for maximum modularity. The three main stages are preprocessing, feature generation, and learning. In the preprocessing stage, pdfs are converted to other formats which can then be used for feature generation. In the feature generation stage, the PDFs themselves along with any results from the preprocessing stage are analyzed for information which might be useful in distinguishing between different scholarly article versions. Lastly, in the learning stage, a machine learning algorithm is applied to the features created in the feature generation stage which creates a scholarly article classifier. Once learning is done, the program stores this classifier so that it can be easily used to make predictions on previously-unseen scholarly articles.  

Since this program is structured as a python package with several submodules, scripts must be run from the root directory of the package with the -m flag provided. For example, to run the preprocessing script pdf\_to\_img.py, you would enter the following command into the shell:
```
python -m preprocessing.pdf_to_img
```

The rest of this README after the dependencies is separated out into five sections which correspond to how to store the PDFs, the three stages of learning, and the prediction process. 

##Dependencies
This program has the following requirements/dependencies:

* A Unix-based machine. It was developed and tested on Ubuntu 14.04 but should work on any Unix-based machine. To make it work on a Windows machine, you would need to change the path names throughout the program to be Windows style or OS-independent. 

* The utility pdftoppm. This program was developed and tested with pdftoppm version 3.03.

* Python 2.7. I don't see why this wouldn't work with 2.6, but I believe some changes would be necessary for it to work with Python 3.

* PDFMiner (developed and tested with build number 20140328). To install this, run

```
sudo easy_install pdfminer
```
If you don't have easy\_install, you can install it with
```
sudo apt-get install python-setuptools
```

* PyPDF2 (a Python package). The program was developed and tested with PyPDF2 version 1.25.1. To install this, run

```
sudo easy_install pypdf2
```

* numpy (tested with version 1.9.2)

* scipy (tested with version 0.15.1)

To install numpy and scipy, you can use
```
sudo apt-get install python-numpy python-scipy
```

* scikit-learn (tested with version 0.17.1). Once you have installed numpy and scipy, you can install scikit-learn with

```
sudo pip install -U scikit-learn
```

#PDF Storage Interface
This program needs to interface with your data (in the form of PDF files) in order to be useful. The interface between your data and the rest of the program is contained in the storage/ directory in a few key files and scripts. 

If you only wish to use the pre-trained model provided to classify your PDFs, you will not need to do very much work to connect your data with the prediction script. See the last section on PDF prediction for information about this.

If, on the other hand, you wish to train your own model from the ground up using your own set of labeled PDFs, you will need to deal with the interface defined in the storage/ directory directly. 

There are two ways you can connect your data with this program:

1. Save your PDFs in a format which the existing storage/article\_store.py file will understand. This means that you will need to put all of your PDFs in the directory storage/pdfs, and each PDF will need to have a filename of the form `<articleid>_<versionid>.pdf`, where articleid is a string which is unique to that article and versionid is 0 if the article is a publisher's version and is 1 if the article is an author's manuscript. More information is available in the comments in storage/article\_store.py. The advantage of this option is that there is no need to modify the code in storage/article\_store.py, but storing your data in this fashion may be impractical, in which case you will need to use option 2.

2. Modify the file storage/article\_store.py so that it properly reads in your data (whether it be stored in a hierarchical directory structure, a database, across a network, or elsewhere). Information about how to do this can be found in the subsection below entitled storage/article\_store.py and in the comments in that file.

Once your data and storage/article\_store.py are talking to each other properly, you will need to generate a set of holdout data (test data) which is not used when training the model. Separating out a test set like this will allow you to assess the accuracy of your trained model objectively without reference to the PDFs which the model was trained on. Information on how to do this can be found in the subsection on storage/generate\_test\_set.py and in the comments in that file. 

Lastly, you may also want to include any PDF names in the file storage/blacklist.csv if you find some PDFs cause the program trouble and you wish to exclude them from all computations. You may have no need to do this, but I found when running this on Harvard data that there were some PDFs which made the PDF parsing utilities this program uses hang. Information on the format of storage/blacklist.csv can be found in the subsection on that file below.

####storage/article\_store.py
This file defines the interface between your data and this program. You may need to change it to recognize your particular data format.  

The purpose of each function is described in the comments in the file, but here are some overall considerations:

* Loading the article ids and versions for all your data from disk is likely to be slow. As such, you only want to perform this load once. This can be achieved by putting the code which accesses disk in storage/article\_store.py outside of any function. When other scripts import the file, this code will run, thereby initializing the article store and allowing the functions defined in the file to run quickly.

* You can test your implementation of the functions in this file by running the python interpreter from the command line in the root directory of this project and then typing `from storage import article_store`. You can then run the functions defined in this file by typing `article_store.function_name()`.  

####storage/generate\_test\_set.py
This script randomly chooses a preset number of PDFs of each class from your data and designates those PDFs as being part of the test set by placing them (each on one line) in the file storage/test\_set.csv. This script should be the first script you run, and once you have trained a model, you can test the model's accuracy on the test set by running the prediction script (described in the last section) on those test articles.

####storage/blacklist.csv
You may encounter PDFs which cause one of the PDF parsing utilities used by this program to crash or hang. This program tries to handle all errors that the PDF parsing utilities might throw, but in case some PDFs manage to crash spectacularly enough to evade this program's error handling, you will probably want to add them to blacklist.csv. The format of blacklist.csv is simple: each line should contain one article\_id that you want to blacklist.

#Preprocessing
The preprocessing scripts are used to turn your pdfs into other formats which can then be used for feature generation. We provide two preprocessing scripts, both of which operate only on the first page of inputted PDFs. The first script, located at preprocessing/pdf\_to\_img.py, turns the first page of the PDF into a color image. The second script, located at preprocessing/pdf\_to\_text.py, extracts the text from the first page of the PDF and stores it in a plaintext file.

####preprocessing/pdf\_to\_img.py
This script converts the first pages of your PDFs into images. Currently, 20dpi images are created, but this can be changed by altering the value of the dpi variable in the script. The script outputs errors to stderr and a progress bar to stdout, so it is recommended that you redirect stderr to a log file or to /dev/null when running this script. This can by done by running 
```
python -m preprocessing.pdf_to_img 2> preprocessing/pdf_to_img.log
```
The program pdftoppm is used to do the actual conversion from PDF to image, and typically there will be some PDFs that it is unable to parse. These PDFs will generate the errors which are sent to stderr.

####preprocessing/pdf\_to\_text.py
This script converts the first pages of your PDFs into plaintext. The plaintext extracted from each PDF is stored in the directory preprocessed/first\_pages in a file called article\_id.txt. Like pdf\_to\_img, this script generates errors when it is unable to parse a particular PDF, and these errors are sent to stderr. See the previous section for an example of redirecting these errors to a log file.

#Feature Generation
The feature generation scripts are the scripts which extract information out of the PDFs in your data set (or out of the preprocessed versions of those PDFs). This step is necessary because machine learning algorithms can't operate very easily on raw PDFs but rather need to deal with simple numbers (called features) which describe various properties of the PDFs. Each script in the feature\_generation directory extracts a particular kind of feature from the PDFs (or preprocessed versions thereof). 

* Word features are features based on what words occur in the first page of the PDF.

* Image features are features based on what the first page of the PDF visually looks like.

* Metadata features are features based on the metadata contained in the PDF file's document information dictionary.

Each script outputs the results (i.e. the calculated features) to a file located under the features/ directory in the subdirectory associated with the type of feature being generated. For word features and image features, the output is a simple CSV file where the rows are article IDs and the columns are the features. For metadata features, the output is a collection of CSV files stored in their own subdirectory, each of which contains information about a particular metadata field.

For word features and metadata features, the name of the output file or folder depends on the parameters used to generate the features. For image features, you can change the name of the output file by changing the value of `feature_file_path` near the end of the file.

####feature\_generation/calculate\_word\_features.py
The idea behind word features is that certain words are more likely to appear in publisher versions than in author manuscripts (and vice versa). This script counts how many times words appear in each article and figures out which of these counts are best at distinguishing between author manuscripts and publisher versions. For example, if the word "creative" consistently appears more often in author manuscripts than in publisher versions, then the script will identify that word as being good at distinguishing between the two classes.

Before running this script:

* Make sure you have run the preprocessing/pdf\_to\_text.py script to generate the plaintext versions of the articles.

* Check if you have a file features/words/feature\_top\_words.csv. If so, it will be overwritten when this script runs.

The output of this script is in the form of a CSV file called feature\_top\_words.csv located in the features/words directory. The columns of this CSV file are words which the script determined to be good at distinguishing between the two classes, and the rows are all the train article ids with valid versions which had been preprocessed into plaintext successfully. Each cell (i,j) contains the count of how many times the jth word appeared in the ith article. If the count is zero, then the cell is empty. The columns are ordered with the ones the script determined to be best at distinguishing between the two article classes appearing farthest to the left.

Parameters you might want to change:

* The script only considers words which appear with some minimum frequency. If the total number of times that a word appears in any article is less than the variable `word_count_cutoff`, the script will discard that word.

* The script will output counts for N top words which are found to be best at distinguishing between the two article classes, where the value of N is stored in the variable `num_words_to_use`. 

####feature\_generation/calculate\_img\_features.py
Image features are features which are calculated based on an image of the first page of the article created by the preprocessing/pdf\_to\_img.py preprocessing script. You can easily add more features by editing the calculate\_features function. Currently, there are several global features being calculated (like the average pixel value, the number above certain cutoff values, etc.) and some local features (like what the longest horizontal and vertical black lines are, whether the page seems to be divided into columns, etc.).

By default, the script looks for images generated at 20dpi. If you generate images with a different dpi in the preprocessing step, you will need to change the value of the `dpi` variable in this script so that it calculates image features based on the images you want it to use.

The output of the script goes to a file whose name is determined by the `feature_file_path` variable. The columns are the features which were calculated and the rows are articles. 

####feature\_generation/calculate\_metadata\_features.py
These features are calculated based on the metadata associated with each article PDF. The PDF format allows documents to contain a document information dictionary -- a metadata store that often contains fields such as the program that was used to generate the PDF, the title of the document, and the document's author. As it turns out, scholarly articles that were typeset by a publisher tend to have characteristically different metadata fields than scholarly articles that were typeset by an individual author. In order to capture this difference in a way that can be used by a machine learning algorithm, we create features representing the number of times that certain words appear in certain metadata fields.

In particular, we first eliminate from consideration metadata fields that don't occur in very many PDFs. Then, for each field that remains, we figure out which words occur most commonly in those fields across all PDFs. We create one feature for every top word for every metadata field. The value of this feature for any given article is the number of times that the word occurs in the metadata field for that particular PDF.

There are two main tunable parameters for this script. The first is the cutoff of how prevalent a metadata field must be in the corpus of PDFs in order for us to make features for that field. The second is the number of features we generate for every field which passes the cutoff (i.e. how many top words we use for each field). The values for these parameters are stored in the variables `field_min_percent_occurrence` and `num_top_words`, respectively. Setting the cutoff to be very low and the number of top words to be very high makes the script take longer and generate more features, but you will likely see decreasing marginal returns as you add increasingly uncommon metadata fields and more and more top words.

The output of this script is more complicated than the output of the other scripts. Instead of putting all the features into a single file, we split up the features so that each metadata field gets its own CSV whose columns are the top words for that metadata field. In addition, we put these feature files into separate folders depending on the parameters under which the script was run. The reason for this is that you might want to generate multiple sets of features with different parameters to measure the tradeoff between using more features to get better performance and using fewer features to get better speed.

#Model Training
Once you have generated features for your training set, you are ready to train the machine learning model. 

The code to train the model is in model.py, but before running it, you may need to make some modifications:

* Depending on what features you generated, you may need to change the variables `metadata_features_specs`, `word_features_specs`, and `img_features_specs`. The values of these variables should match the corresponding value used to create the features. For metadata features, this is the `metadata_spec` variable in feature\_generation/calculate\_metadata\_features.py. For word features, it is the variable `word_feautres_specs` in feature\_generation/calculate\_word\_features.py. For image features, it is the variable `img_features_specs` in feature\_generation/calculate\_img\_features.py.

* There are two flags that control the output of the script:

    * `save_model` is a flag that controls whether the trained model is saved in the trained\_models directory. If set to true, the model is saved in a file with the same name as the variable `model_name`.

    * `write_performance` is a flag that controls how the script outputs the cross-validated performance that the model achieved on the training data. If set to true, the performance is written to the file results/performance.csv. If set to false, the performance is written to stdout. In the former case, the file output includes a human-readable description of the model stored in the variable `features_description`.

* By default, the script uses logistic regression to learn from the features. If you want to use a different model, you can change the value of the `clf` variable. Make sure to read the comment about the `class_weight` parameter before trying out a different model. (If you change the model, you may also want to change the name of the model that is written to the performance file when `write_performance` is set to true.)

* The variable `xval` controls the number of folds used for cross-validation.

The script takes as input the required confidence that the model must have before making a prediction. If set to 0.5, the model should make predictions for every input. To run the script:
```
python -m model <required_confidence>
```

#Prediction
Once you have trained a model, you can use the model to predict the versions of previously unseen PDFs. The relevant script is predict/predict\_articles.py. This script does the following:

1. Runs the preprocessing scripts on all the PDFs stored in the directory stored in the variable `predictee_folder`.

2. Runs the feature generation scripts on the articles using the results of the preprocessing using the specs specified by the variables `metadata_feature_specs`, `word_feature_specs`, and `img_feature_specs`.

3. Loads the model named `model_name` and runs it on the resulting features.

4. Writes the predicted labels to `output_folder`. Note that the labels are outputted as a real number between 0 and 1 representing the model's confidence about the assigned label. To recover the predicted label, simply round the output or use `clf.predict(X)` instead of `clf.predict_proba(X)`.
