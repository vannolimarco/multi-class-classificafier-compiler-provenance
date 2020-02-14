from sklearn.naive_bayes import GaussianNB
import preprocessing
import pathconfig
import model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import evaluation_metrics
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import *
from sklearn import metrics
import pickle
'''
###########################################         MAIN        ################################################
this file py has the purpose to use all methods need to obtain the followings scopes:
  - make a preprocessign to dataset provided for training and set: binary and multi-class classification problems,
  - create the models based on different kind of algorithms: they can be instantiated by different
    sub-class provided by the class Model implemented. The models can be chosen in order to test 
    for both classification problems provided, i.e binary and multi-class 
  - split the features and targets in order to obtain a balance input and output for training of mode,
  - perform the training of model chosen,
  - save the models,
  - perform a prediction and save it in a file in format csv,
  - evaluate the prediction with the test set provided (blind):
      - compute and plot the confusion matrix,
      - print the report of classification,
      - compute and print the accuracy of models,
  ####################################################################################################
'''

'''
Preprocessing:
In this section the preprocessing is performed:
- obtain the features with two kind of preprocessing adopted: unigrams and bigrams with occurence,
- the models are instanced with different kind of algorithm availabled in class 'Model'
'''
Preprocessing_data = preprocessing.Preprocessing()                 #instance of preprocessing data
Preprocessing_prediction = preprocessing.PreprocessingPrediction() #instance of preprocessing for prediciton phase

kind_preprocessing_unigrams = Preprocessing_data.unigrams #kind of preprocessing unigrams
kind_preprocessing_bigrams = Preprocessing_data.bigrams   #kind of preprocessing bigrams
kind_label_opt = Preprocessing_data.opt                   #kind of label used for binary class: opt
kind_label_compiler = Preprocessing_data.compiler         #kind of label used for multi-class class: compiler

'''
The choice of Algorithm:
In oder to use the algorithm in main is need to instance the desidered algorithm 
from the following list:
- Naive Bayes Multinomial algorithm => NaiveBayesMultinomial() 
- Naive Bayes Bernoulli algorithm => NaiveBayesMBernoulli()
- SVC algorithm with 'rbf' as kernel => RbfSVC()
- SVC algorithm with 'Linear' as kernel => LinearSVC()
- Decision Tree algorithm => DecisionTree()
'''
Model_binary = model.LinearSVC()                #instance of model for binary classification task
Model_multi_class = model.LinearSVC()           #instance of model for multi-class classification task

print('The algorithms: \n')
print('- Binary Classification problem (Hight, Low): {}'.format(Model_binary.name))
print('- Multi-Class Classification problem (gcc,clang,icc): {}'.format(Model_multi_class.name))

train_path_dataset = Preprocessing_data.path_train_dataset    #the path of dataset input
test_path_dataset = Preprocessing_data.path_test_dataset      #the path of dataset test blind
precition_path = Preprocessing_data.path_prediction           #the path of prediction

'''
The choice of Preprocessing:
create the features for training:
  - In order to choose the preprocessing is need to put in the parameter 'kind_preprocessing',
    'kind_preprocessing_unigram' if want to use unigrams preprocessing and  'kind_preprocessing_bigrams'
    if want to use bigrams preprocessing with occurrences (CountVectorizer(ngram_range=(1,2))),
  - the parameter 'train' is true because it is used for training here.
'''
features = Preprocessing_data.get_features(path_file_json= train_path_dataset, train=True,kind_preprocessing=kind_preprocessing_bigrams) #create features,

# create the labels for class_ gcc, clang and icc.
output_classes = Preprocessing_data.get_labels(path_file_json=train_path_dataset, kind_label=kind_label_compiler)

# create the labels for binary: Hight/Low (1/0)
output_opt = Preprocessing_data.get_labels(path_file_json=train_path_dataset, kind_label=kind_label_opt)

# Split dataset into training set and test set for multi-class classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(features, output_classes, test_size=0.3,random_state=0) # 70% training and 30% test


# Split dataset into training set and test set for binary cassification
X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(features, output_opt, test_size=0.3,random_state=0) # 70% training and 30% test

'''
Training of models:
In this section the training of models is performed:
- training of models,
- save models.
'''

# Train the model for binary classification task using the training sets
model_binary = Model_binary.fit_model(X_train_opt,y_train_opt)

# Train the model for multi-class classification task  the training sets
model_multi_class = Model_multi_class.fit_model(X_train_class,y_train_class)

# save the binary model
Model_binary.save_model(path_file_save=Model_binary.PATH_SAVE_MODEL_BINARY, model=model_binary)

# save the multi-classification model
Model_multi_class.save_model(path_file_save=Model_multi_class.PATH_SAVE_MULTI_CLASS, model=model_multi_class)


'''
Prediction:
In this section the prediction is performed:
- creating of the features for prediction from blind file provided,
- performing of prediction from fetaures,
- save the prediction in csv file,
'''

# create the features for prediction:
# -choose the preprocessing -> kind_preprocessing = kind_preprocessing_unigram or kind_preprocessing_bigrams,
# -the parameter 'train' is false because it is used for prediction here.
features_prediction = Preprocessing_data.get_features(path_file_json=test_path_dataset, train=False, kind_preprocessing=kind_preprocessing_bigrams)

predition_opt = []        #array for binary prediction
prediction_compiler = []  #array for multi-classification prediction

#cycle on features for prediciton in order to predict singlse feature for both binary and multi-classification problems.
for feature in features_prediction:
    predition_opt.append(model_binary.predict(feature))
    prediction_compiler.append(model_multi_class.predict(feature))

# convert the prediction from numbers to labels for 'opt', so 'H' for Hight from the value 1 and so 'L' for Low from the value 0
prediction_opt = Preprocessing_prediction.convert_labels(predictions=predition_opt,kind_label=kind_label_opt)

# convert the prediction from numbers to labels for 'compiler', so 'gcc' from the value 0, 'clang' form value 1 and 'icc' from value 2.
prediction_compiler = Preprocessing_prediction.convert_labels(predictions=prediction_compiler,kind_label=kind_label_compiler)

# join the the two arrays of prediction in order to obtain a singlse array.
prediction = zip(prediction_compiler,prediction_opt)

# save the prediction in a csv file.
Preprocessing_prediction.save_prediction_csv(path_file_save=precition_path,predictions=prediction)

'''
Evaluation:
In this section the model is evaluated through the confusion matrix and accuracy.
'''

# perform of prediciton for binary classification problem.
y_pred_opt = model_binary.predict(X_test_opt)

# evaluate and plot the confusion matrix normalized for binary classification problem
evaluation_metrics.plot_confusion_matrix_norm_binary(y_test_opt, y_pred_opt)

# evaluate and plot the classification report for binary classification problemù
print('classification report for binary classification task: \n')
evaluation_metrics.print_classification_report(y_test_opt,y_pred_opt)

# perform of prediciton for multi-class classification problem.
y_pred_class = model_multi_class.predict(X_test_class)

# evaluate and plot the confusion matrix normalized for multi-class classification problem
evaluation_metrics.plot_confusion_matrix_norm_multi_class(y_test_class, y_pred_class)

# evaluate and plot the classification report for multi-class classification problem
print('classification report for multi-class classification task: \n')
evaluation_metrics.print_classification_report(y_test_class, y_pred_class)

# compute and print the accuracy for model about binary classification problem
print("Accuracy Model Binary Classification : {}".format(evaluation_metrics.accuracy_score(y_test_opt, y_pred_opt)))

# compute and print the accuracy for model about multi-class classification problem
print("Accuracy Model Multi-class Classification: {}".format(evaluation_metrics.accuracy_score(y_test_class, y_pred_class)))