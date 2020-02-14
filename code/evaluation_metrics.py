from matplotlib import pyplot as plt
import  pandas as pd
import  seaborn as sn
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
'''
This file py aims to expose methods in order to obtain an evaluation for results given by model.
'''

def plot_confusion_matrix_norm_binary(y_test,y_pred):
    '''
    this method has the purpose to plot the confusion matrix for binary classification problem. It takes as parameter the
    the targets from testing set and the target predicted by model.
    '''
    array = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix binary classification task: \n{}'.format(array))
    array_norm =  np.divide(array, 9000)
    df_cm = pd.DataFrame(array_norm, index = [i for i in ['low','higth']],columns = [i for i in  ['low','hight']])
    sn.set(font_scale=1.3)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 20})  # font size
    plt.title('Multi-Class Classification Confusion Matrix}')
    plt.show()

def plot_confusion_matrix_norm_multi_class(y_test,y_pred):
    '''
    this method has the purpose to plot the confusion matrix for multi-class classification problem. It takes as parameter the
    the targets from testing set and the target predicted by model.
    '''
    array = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix Multi-class classification task: \n{}'.format(array))
    array_norm = np.divide(array, 9000)
    df_cm = pd.DataFrame(array_norm, index = [i for i in ['gcc','clang','icc']],columns = [i for i in  ['gcc','clang','icc']])
    sn.set(font_scale=1.3)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 20})  # font size
    plt.title('Multi-Class Classification Confusion Matrix')
    plt.show()

def print_classification_report(y_test,y_pred):
    '''
    this method has the purpose to compute and then print the classification report form targtes test and targets predicted.
    '''
    return print(classification_report(y_test, y_pred))

def accuracy(y_test,y_pred):
    '''
       this method has the purpose to compute and then print the accuracy form targtes test and targets predicted.
    '''
    return print(accuracy_score(y_test, y_pred))