#MODEL
#Import Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import preprocessing
import pathconfig
from sklearn.model_selection import train_test_split
import pickle
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV

# defining parameter range

paths = pathconfig.paths()

class Model:
      '''
         Class to define model in different problem fo classification: binary and multi-class.
         it is formed by two sub-class: one for binary classification and one for multi-class classification.
         P.S methods fit_model(x_train,y_train) aims to fit the model instantiated by sub.class and return it. It takes as parameters
         x_ and y from training set.
      '''
      def __init__(self):
          self.PATH_SAVE_MODEL_BINARY = paths.PATH_MODEL_BINARY_CLASSIFICATION
          self.PATH_SAVE_MULTI_CLASS = paths.PATH_MODEL_MULTI_CLASS_CLASSIFICATION

      def save_model(self,path_file_save:str, model:object) -> None:
            '''
                this method aims to save the model just created. It takes as parameter the path of file where save the model.
            '''
            try:
                with open(path_file_save, 'wb') as file:
                    pickle.dump(model, file)
                    return print('Model[%s saved in file] %s' %(model,path_file_save))
            except Exception as error:
                print(error)

class NaiveBayesMultinomial(Model):
    '''
        Sub-class to implement the model based by Naive Bayes Multinominal algorithm.
    '''

    def __init__(self):
        self.name = 'Naive Bayes Multinomial'
        self.PATH_SAVE_MODEL_BINARY = paths.PATH_MODEL_BINARY_CLASSIFICATION
        self.PATH_SAVE_MULTI_CLASS = paths.PATH_MODEL_MULTI_CLASS_CLASSIFICATION
    def fit_model(self, x_train, y_train) -> object:
        model = MultinomialNB()
        model.fit(x_train, y_train)
        return model

class NaiveBayesBernoulli(Model):
    '''
        Sub-class to implement the model based by Naive Bayes Bernoulli algorithm.
    '''

    def __init__(self):
        self.name = 'Naive Bayes Bernoulli'
        self.PATH_SAVE_MODEL_BINARY = paths.PATH_MODEL_BINARY_CLASSIFICATION
        self.PATH_SAVE_MULTI_CLASS = paths.PATH_MODEL_MULTI_CLASS_CLASSIFICATION

    def fit_model(self, x_train, y_train) -> object:
        model = BernoulliNB()
        model.fit(x_train, y_train)
        return model


class RbfSVC(Model):
      '''
            Sub-class to implement the model based by SVC algorithm with rbf kernel.
      '''

      def __init__(self):
          self.name = 'Support Vector Classification (SVC) - rbf kernel'
          self.PATH_SAVE_MODEL_BINARY = paths.PATH_MODEL_BINARY_CLASSIFICATION
          self.PATH_SAVE_MULTI_CLASS = paths.PATH_MODEL_MULTI_CLASS_CLASSIFICATION

      def fit_model(self, x_train, y_train) -> object:
          clf = svm.SVC(C=1,kernel= 'rbf', degree = 3, gamma =0.001, coef0 = 0.0, shrinking = True, probability = False, tol = 0.001, cache_size = 300, class_weight = None, verbose = False, max_iter = -1, decision_function_shape ='ovo', random_state = None)
          model = clf.fit(x_train, y_train)
          return model

class LinearSVC(Model):
    '''
        Sub-class to implement the model based by SVC algorithm with Linear kernel.
    '''

    def __init__(self):
        self.name = 'Support Vector Classification (SVC) - Linear kernel'
        self.PATH_SAVE_MODEL_BINARY = paths.PATH_MODEL_BINARY_CLASSIFICATION
        self.PATH_SAVE_MULTI_CLASS = paths.PATH_MODEL_MULTI_CLASS_CLASSIFICATION

    def fit_model(self, x_train, y_train) -> object:
        clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=2000)
        model = clf.fit(x_train, y_train)
        return model

class DecisionTree(Model):
       '''
            Sub-class to implement the model based by Decision Tree algorithm.
       '''
       def __init__(self):
           self.name = 'Decision Tree'
           self.PATH_SAVE_MODEL_BINARY = paths.PATH_MODEL_BINARY_CLASSIFICATION
           self.PATH_SAVE_MULTI_CLASS = paths.PATH_MODEL_MULTI_CLASS_CLASSIFICATION

       def fit_model(self, x_train, y_train) -> object:
         dtree_model = DecisionTreeClassifier(max_depth=2).fit(x_train, y_train)
         return dtree_model


