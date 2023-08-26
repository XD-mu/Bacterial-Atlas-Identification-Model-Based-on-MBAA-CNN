import os
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from pretty_confusion_matrix import pp_matrix_from_data
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.metrics import precision_recall_curve, average_precision_score
accuracy=0.675
labels=['MRSA','A' ,'B', 'C', 'D', 'E', 'F', 'G','H', 'I']
y_test=[7 ,4 ,7 ,7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,0, 0, 0, 0,
 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 ,2, 2, 2, 2, 2, 2, 2, 2, 2,
 2 ,2 ,2 ,2 ,2, 2, 3, 3, 3, 3 ,3 ,3, 3 ,3, 3, 3, 3,3 ,3, 3, 3 ,3, 3, 3,3 ,3 ,4, 4, 4 ,4, 4 ,4 ,4 ,4 ,4, 4, 4,
 4 ,4 ,4, 4 ,4, 4, 4, 4, 4 ,5 ,5, 5, 5, 5 ,5 ,5, 5 ,5, 5,5 ,5, 5 ,5 ,5, 5 ,5 ,5, 5 ,5,6 ,6, 6,6 ,6 ,6 ,6, 6,
 6 ,6 ,6 ,6 ,6 ,6 ,6 ,6 ,6 ,6 ,6 ,6,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]
y_pred_classes=[6 ,6 ,7 ,7, 8, 8, 5, 6, 4, 6, 7, 6, 6, 7, 6, 7, 6, 6, 6, 7, 1, 0, 0, 0, 1, 0,1, 1, 0, 2 ,0 ,0,0 ,0, 0, 0, 0,
 0, 0, 0, 1, 1, 1, 1, 2, 3, 5, 1, 3, 5, 1, 4, 1, 5, 2, 3, 1, 1, 1, 1, 2, 2, 3, 3, 2 ,2, 2, 2, 2, 2, 2, 2, 2, 2,
 2 ,2 ,2 ,2 ,2, 2, 3, 3, 3, 3 ,3 ,3, 3 ,3, 3, 2, 3,3 ,3, 3, 3 ,3, 3, 3,3 ,3 ,4, 4, 4 ,4, 4 ,4 ,4 ,4 ,4, 4, 4,
 4 ,4 ,4, 4 ,4, 4, 4, 4, 4 ,5 ,5, 4, 5, 5 ,5 ,5, 5 ,5, 5,5 ,5, 5 ,5 ,5, 5 ,6 ,5, 5 ,6,6 ,6, 6,6 ,6 ,6 ,6, 6,
 3 ,5 ,6 ,5 ,5 ,5 ,5 ,5 ,5 ,6 ,8 ,6,8,8,6,7,8,7,8,7,8,4,8,4,8,3,4,5,4,4,3,8,9,8,3,9,9,9,9,9,6,9,7,8,6,7,7,7,9,7,9,9]


cmap = "PuRd"
pp_matrix_from_data(y_test, y_pred_classes,columns=labels,lw=accuracy,cmap=cmap)

