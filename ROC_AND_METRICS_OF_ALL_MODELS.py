#import keras
import tensorflow as tf
import numpy as np
#from sklearn.metrics import classification_report
#import pydicom as dcm
import os
import numpy as np
#import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from segment_brain import segment
from tqdm import tqdm
import re
from segment_brain import segment_all_patients_slices
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from My_3Dmodel_AUC import MultipleInputsModel_3D
# from CT_DATASET_module_with_Classes_rescale import *
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage
import random
# from augmentationsMINORITY import CT_augmentations
# from sklearn.utils.class_weight import compute_class_weight
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping
#import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score,recall_score,precision_score,balanced_accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3,4,5,6,7"
strategy = tf.distribute.MirroredStrategy()

patients1 = np.load("/raid/theodoropoulos/PhD/Data/128x128x120/Train/patients_Train.npy",allow_pickle=True)
patients2 = np.load("/raid/theodoropoulos/PhD/Data/128x128x120/Test/patients_Test.npy",allow_pickle=True)

patients = np.concatenate((patients1,patients2),axis=0)
del patients1,patients2

np.random.seed(42)
np.random.shuffle(patients)

X = np.array([patients[i]['volume']  for i in range(len(patients)) ])
X =np.transpose(X,(0,2,3,1))
X = segment_all_patients_slices(X)
print(X.shape)
y = np.array (  [patients[i]['Class']  for i in range(len(patients)) ]).astype('int32')

#######################
labels_sex = np.array([patients[i]['sex']  for i in range(len(patients)) ])
le = LabelEncoder()
le.fit(labels_sex)
labels_sex_transf = le.transform(labels_sex)

######################

labels_age = np.array([patients[i]['age']  for i in range(len(patients)) ])
labels_age_categ = []
for age in labels_age:
    if age=='NA':
        labels_age_categ.append('NA')
    elif int(age)<30:
        labels_age_categ.append('Adult')
    elif int(age)>=30 and int(age)<60:
        labels_age_categ.append('Middle')
    else:
        labels_age_categ.append('Old')

labels_age_categ =np.array(labels_age_categ)

le = LabelEncoder()
le.fit(labels_age_categ)
labels_age_transf = le.transform(labels_age_categ)
####################
labels_GCS = np.array([patients[i]['Glasgow Coma Scale']  for i in range(len(patients)) ])
labels_GCS_categ = []
for GCS in labels_GCS:
    if GCS=='NA':
        labels_GCS_categ.append('NA')
    elif int(GCS)<=8:
        labels_GCS_categ.append('HIGH')
    else:
        labels_GCS_categ.append('LOW')

labels_GCS_categ =np.array(labels_GCS_categ)

le = LabelEncoder()
le.fit(labels_GCS_categ)
labels_GCS_transf = le.transform(labels_GCS_categ)

##############################

X_train,  X_val,  y_train, y_val,   labels_sex_train, labels_sex_val,  labels_age_train, labels_age_val ,label_GCS_train, label_GCS_val= train_test_split( X, y,
                                                                                                                                                                labels_sex_transf,
                                                                                                                                                                labels_age_transf,
                                                                                                                                                                labels_GCS_transf,
                                                                                                                                                                test_size=0.3,random_state=42)


X_val,  X_test,  y_val, y_test,   labels_sex_val, labels_sex_test,  labels_age_val, labels_age_test ,label_GCS_val, label_GCS_test= train_test_split( X_val, y_val,
                                                                                                                                                                labels_sex_val,
                                                                                                                                                                labels_age_val,
                                                                                                                                                                label_GCS_val,
                                                                                                                                                                test_size=0.1,random_state=42)

batch_size = 8
def validation_preprocessing(volume, labels_sex, labels_age, labels_gcs, y):

    volume = tf.expand_dims(volume, axis=3)
    return (volume, labels_sex, labels_age, labels_gcs), y

validation_loader = tf.data.Dataset.from_tensor_slices((X_val,    
                                                     labels_sex_val, 
                                                     labels_age_val, 
                                                     label_GCS_val, 
                                                     y_val))

validation_dataset = (
    #validation_loader.shuffle(len(X_val))
    validation_loader.map(validation_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)



print("loaded dataset")
# Paths to the saved models
model_paths = {
  "Custom 3D": "/raid/theodoropoulos/PhD/Results/whole image/128_128_120/3D/3D_full_image_model_120_128_128_Tensorflow_16.keras",
   "DenseNet201": '/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/DenseNet201_120_128_128.keras',
    "ResNet101": '/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/Resnet101_120_128_128.keras',
 "MobileNetV2": '/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/MobileNetV2_120_128_128.keras',
#   "Seresnet18": '/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/Seresnet18_120_128_128.keras'
}


print("loaded paths")
# Dictionary to store the results
results = []

# Plot ROC curves for all models
plt.figure(figsize=(10, 8))

del patients
for model_name, model_path in model_paths.items():
    # Load the model
    model = tf.keras.models.load_model(model_path,compile=False)
    print("Loading....:{}".format(model_name))
    # Predict the validation set
    preds = model.predict(validation_dataset)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_val, preds)
    auc_value = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_value:.2f})')

    # Compute Youden's J statistic for each threshold
    j_scores = tpr - fpr
    optimal_idx_roc = np.argmax(j_scores)
    optimal_threshold_roc = thresholds[optimal_idx_roc]

    # Confusion matrix
    y_pred = (preds > optimal_threshold_roc).astype('float')
    tn, fp, fn, tp = confusion_matrix(y_true=y_val, y_pred=y_pred).ravel()

    # Calculate metrics
    f1 = f1_score(y_true=y_val, y_pred=y_pred,average='weighted')
    recall = recall_score(y_true=y_val, y_pred=y_pred,average='weighted')
    precision = precision_score(y_true=y_val, y_pred=y_pred,average='weighted')
    accuracy = balanced_accuracy_score(y_true=y_val, y_pred=y_pred)
    # recall = tp / (tp + fn)
    # precision = tp / (tp + fp)
    # accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Append the results to the list
    results.append({
        "Model": model_name,
        "AUC": auc_value,
        "F1 Score": f1,
        "Recall": recall,
        "Precision": precision,
        "Accuracy": accuracy
    })

    del model,preds

# Finalize the ROC curve plot
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc='lower right')
plt.grid(True)

# plt.savefig('/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/ALL_ROCs_full.png')

print(results)






















