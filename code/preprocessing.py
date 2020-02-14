import pathconfig
import jsonlines
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
import pandas as pd
import itertools
from sklearn.feature_extraction.text import *
import csv
import re
from itertools import groupby
from sklearn.feature_extraction.text import *

paths = pathconfig.paths()


class Preprocessing:
    '''
     Preprocessing Class for preprocessing phase. It implement a several number of methods used to
     perform some action for preprocessing. It is the main class with which is possible
     call all methods destined to perform preprocessing for training of model.
    '''

    # GLOBAL VARIABLES OF FILE JSONL'S ELEMENTS OF SINGLE FUNCTION
    def __init__(self):
        self.path_train_dataset = paths.PATH_TRAIN_DATASET  #the path of training dataset
        self.path_test_dataset = paths.PATH_TEST_DATASET  #the path of test dataset
        self.path_prediction = paths.PATH_PREDICTION_CVS #the path where will save the prediction
        self.instruction = 'instructions'
        self.opt = 'opt'
        self.compiler = 'compiler'
        self.class_gcc = 'gcc'
        self.class_clang = 'clang'
        self.class_icc = 'icc'
        self.label_compiler = [0,1,2]   #gcc,clang,icc class
        self.opt_H = 'H'                #opt H target
        self.opt_L = 'L'                #opt L target
        self.label_opt_H = 1
        self.label_opt_L = 0
        self.unigrams = 'unigrams'
        self.bigrams = 'bigrams'


    def get_all_instructions_opt_comp(self, path_file_json: str) -> object:
        '''
               this method return all instruction with mnemonics, opt and compiler targets.
               It is used to obtain the output about binary and multi-class classification
               task. Take as parameters the path of dataset in json format.
        '''
        with jsonlines.open(path_file_json) as reader:
            list_funtions = []
            for function in reader:
                new_function = []
                instrunction = function[self.instruction]    #instruction
                for instr in instrunction:
                    new_function.append(instr.split()[0])    #olny first word of binary code
                new_function.append(function[self.opt])      #opt
                new_function.append(function[self.compiler]) #compiler
                list_funtions.append(new_function)
            return list_funtions

    def get_all_instructions_mnemonics(self, path_file_json: str, kind_preprocessing: str) -> object:
        '''
           this method return all instruction with only mnemonics.
           It is used to obtain the input about binary and multi-class classification
           task. Take as parameters the path of dataset in json format.
        '''
        with jsonlines.open(path_file_json) as reader:
            if(kind_preprocessing == self.unigrams):  #if the preprocessing is unigram
                list_funtions = []
                for function in reader:
                    new_function = []
                    instrunction = function[self.instruction]
                    for instr in instrunction:
                        new_function.append(instr.split()[0])
                    list_funtions.append(new_function)
                return list_funtions
            elif(kind_preprocessing == self.bigrams):  #if the preprocessing is unigram
                list_funtions = []
                new_function = ""
                for function in reader:
                    instrunction = function[self.instruction]
                    for instr in instrunction:
                        new_function += (instr.split()[0]) + ' '
                    list_funtions.append((new_function))
                    new_function = ""
                return list_funtions
            else:
                print('the method accepts only &s and &s as kind of preprocessing.' &(self.unigrams,self.bigrams))

    def get_vocab_mnemonics(self, path_file_jsonl: str) -> object:
        '''
           this method return the vocabulary of each mnemonics. It is used to prepare feature in unigram modality.
           It takes as paraemters teh file of json where it extracts the mnemonics.
        '''
        with jsonlines.open(path_file_jsonl) as reader:
            list_funtions = []
            for function in reader:
                instrunction = function[self.instruction]
                for instr in instrunction:
                    list_funtions.append(instr.split()[0])
            list_funtions = list(set((list_funtions)))
            list_funtions.sort()
            vocabolary = dict()
            vocabolary['UNK'] = 1
            index = 2
            for word in list_funtions:
                vocabolary[word] = index
                index += 1
            return vocabolary

    def get_vocab_compiler(self) -> object:
        '''
              this method return the vocabulary for classes of compilers. They are gcc, clang and icc
              and each of them takes as values 0,1 and 2.
        '''
        list_class = [self.class_gcc,self.class_icc,self.class_clang]
        index = 0
        vocabolary = dict()
        for classe in list_class:
                vocabolary[classe] = index
                index += 1
        return vocabolary

    def get_features(self, path_file_json: str, train:bool, kind_preprocessing:str) -> object:
        '''
             this method returns all the features which need to train the model.It has a compact form because  t split into two preprocessing methods:
             one of the two is unigram where each mnemonics is taken and replaced with its value inside vocabulary; the second is bigrams method performed with
             method CountVectorize that compute the occurrence of the bigrams in each instruction. This method takes as parameters the file of dataset json,
             a boolean which tells us if the feature are for train or for prediction and the kind of preprocessing: unigrams or bigrams.
         '''
        if(kind_preprocessing == self.bigrams):  #if the preprocessing is bigrams
            if(train):
                instructions_input = self.get_all_instructions_mnemonics(path_file_json=path_file_json,kind_preprocessing=kind_preprocessing)
                vectorizer = CountVectorizer(ngram_range=(1,2))
                input_bigrams = vectorizer.fit_transform(instructions_input)
                return input_bigrams
            else:
                instructions_input = self.get_all_instructions_mnemonics(paths.PATH_TRAIN_DATASET, kind_preprocessing=kind_preprocessing)
                vectorizer = CountVectorizer(ngram_range=(1, 2))
                input_bigrams = vectorizer.fit_transform(instructions_input)
                instructions_input_test = self.get_all_instructions_mnemonics(path_file_json=path_file_json,kind_preprocessing=kind_preprocessing)
                vectorizer_test = CountVectorizer(vocabulary = vectorizer.get_feature_names(),ngram_range=(1, 2))
                input_bigrams_test = vectorizer_test.fit_transform(instructions_input_test)
                return input_bigrams_test
        elif(kind_preprocessing == self.unigrams):  #if the preprocessing is unigram
            mnemonics = []
            input = []
            instructions_input = self.get_all_instructions_mnemonics(path_file_json=path_file_json,kind_preprocessing=kind_preprocessing)
            vocabulary = self.get_vocab_mnemonics(path_file_jsonl=paths.PATH_TRAIN_DATASET)
            for instruction in instructions_input:
                for instr in instruction:
                    if (instr in vocabulary):
                        mnemonics.append(vocabulary[instr])
                    else:
                        mnemonics.append(vocabulary['UNK'])
                mnemonics_features = mnemonics[:3] + mnemonics[-3:]
                input.append(np.array((mnemonics_features)))
                mnemonics = []
            input = pad_sequences(input, truncating='pre', padding='post', maxlen=6)
            return input
        else:
            print('the method accepts only &s and &s as kind of preprocessing.' & (self.unigrams, self.bigrams))

    def get_labels(self, path_file_json: str, kind_label=str) -> object:
        '''
             this method returns all the labels or targets which need to train the model.It has a compact form because takes into account both classification
             probelms: one for label 'opt' (binary) and one for 'compiler' (multi-class).The parameters that it accepts are: the path of dataset and
             the kind of label = 'opt' or 'compiler'.
        '''
        if(kind_label == self.opt):        #if the label is for the optimizer
            instructions = self.get_all_instructions_opt_comp(path_file_json=path_file_json)
            list_output = []
            for instr in instructions:
                if (instr[-2] == self.opt_H):
                    list_output.append(self.label_opt_H)
                else:
                    list_output.append(self.label_opt_L)
            return list_output
        elif(kind_label == self.compiler):  #if the label is for the compiler
            instructions = self.get_all_instructions_opt_comp(path_file_json=path_file_json)
            vocabulary= self.get_vocab_compiler()
            list_output = []
            for instr in instructions:
                if (instr[-1] in vocabulary):
                    list_output.append(vocabulary[instr[-1]])
                else:
                    print('the instruction: <%s> not cointains class label' % instr)
                    break;
            return list_output
        else:
            print('the method accepts only &s and &s as labels.' & (self.opt, self.compiler))

class PreprocessingPrediction(Preprocessing):
        '''
        PreprocessingPrediction Class for prediction phase.It is a sub-class of Preprocessing because it aims to perform a reverse
         preprocessing after prediction of model. It implement a several number of methods used to obtain some actions for prediction
         as convertion and save predictions into a file csv.
        '''
        def save_prediction_csv(self, path_file_save: str, predictions:object ) -> None:
            '''
                this method aims to save into a file csv the predictions which the model has obtained.It takes as parameters the path of
                file where save the predicitons and the prediction object returned by model.
            '''
            with open(path_file_save, mode='w') as csv_file:
                fieldnames = [self.compiler, self.opt]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for element in predictions:
                    writer.writerow({self.compiler: element[0],self.opt: element[1]})
                print('The predictions were saved in csv format: [path] -> {}'.format(path_file_save))

        def convert_labels(self, predictions: object, kind_label: str)-> object:
            '''
                  this method has the purpose to convert the 'opt' labels or 'compiler' labels depending on
                  the kind of label which is chosen as the parameter of the method: 'opt' or 'compiler'
            '''
            if(kind_label == self.opt):
                list_opt_converted = list()
                for opt in predictions:
                    if(opt[0] == self.label_opt_H):
                        list_opt_converted.append(self.opt_H)
                    else:
                        list_opt_converted.append(self.opt_L)
                return list_opt_converted
            elif(kind_label == self.compiler):
                list_compiler_converted = list()
                for compiler in predictions:
                    if (compiler == self.label_compiler[0]):          #gcc
                        list_compiler_converted.append(self.class_gcc)
                    elif (compiler == self.label_compiler[1]):        #clang
                        list_compiler_converted.append(self.class_icc)
                    elif (compiler == self.label_compiler[2]):        #icc
                        list_compiler_converted.append(self.class_clang)
                    else:
                        print('there is label with no correct value : %s ' % compiler)
                return list_compiler_converted
            else:
                print('the method accepts only &s and &s as labels.' & (self.opt, self.compiler))

