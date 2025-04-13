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
# from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score,recall_score,precision_score,balanced_accuracy_score
from Compute_Metrics import *

os.environ["CUDA_VISIBLE_DEVICES"] ="6"
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
                                                                                                                                                                test_size=0.3,random_state=42)

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

def test_preprocessing(volume, labels_sex, labels_age, labels_gcs, y):

    volume = tf.expand_dims(volume, axis=3)
    return (volume, labels_sex, labels_age, labels_gcs), y

test_loader = tf.data.Dataset.from_tensor_slices((X_test,    
                                                     labels_sex_test, 
                                                     labels_age_test, 
                                                     label_GCS_test, 
                                                     y_test))

test_dataset = (
    #test_loader.shuffle(len(X_test))
    test_loader.map(test_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)



print("loaded dataset")
# Paths to the saved models
model_paths = {
  "Custom 3D": "/raid/theodoropoulos/PhD/Results/whole image/128_128_120/3D/3D_full_image_model_120_128_128_Tensorflow_16.keras",
   "DenseNet201 3D": '/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/DenseNet201_120_128_128.keras',
    "ResNet101 3D": '/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/Resnet101_120_128_128.keras',
 "MobileNetV2 3D": '/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/MobileNetV2_120_128_128.keras',
}


print("loaded paths")
del patients

del X
import gc
import tensorflow.keras.backend as K

for model_name, model_path in model_paths.items():
    print(f"Model: {model_name}")

    # Create a single figure for both ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Load model
    with strategy.scope():
        model = tf.keras.models.load_model(model_path)

    # Create directory for saving metrics
    save_dir = f'/raid/theodoropoulos/PhD/Results/Revision/Multi_ALL/{model_name}/'
    os.makedirs(save_dir, exist_ok=True)

    # File to save evaluation metrics
    metrics_file = os.path.join(save_dir, f"{model_name} metrics.txt")

    with open(metrics_file, 'w') as f:
        f.write(f"Model: {model_name}\n")

    for dataset, dataset_mode, y in zip([validation_dataset, test_dataset], 
                                        ["Validation", "Test"], 
                                        [y_val, y_test]):
        print(f"Processing {dataset_mode} dataset...")

        # Predict
        preds = model.predict(dataset)

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y, preds)
        auc_val = auc(fpr, tpr)

        # Compute optimal threshold using Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

        # Generate binary predictions
        y_pred = (preds > optimal_threshold).astype('float')

        # Compute Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_true=y, y_pred=y_pred).ravel()

        # Compute additional metrics
        f1 = f1_score(y_true=y, y_pred=y_pred)
        recall = recall_score(y_true=y, y_pred=y_pred)
        precision = precision_score(y_true=y, y_pred=y_pred)
        pr_auc = average_precision_score(y, preds)

        # Save metrics to file
        with open(metrics_file, 'a') as f:
            f.write(f"\nDataset: {dataset_mode}\n")
            f.write(f"AUC: {auc_val:.3f}\n")
            f.write(f"PR AUC: {pr_auc:.3f}\n")
            f.write(f"F1 Score: {f1:.3f}\n")
            f.write(f"Recall: {recall:.3f}\n")
            f.write(f"Precision: {precision:.3f}\n")
            f.write("Confusion Matrix:\n")
            f.write(f"  True Negative: {tn}\n")
            f.write(f"  False Positive: {fp}\n")
            f.write(f"  False Negative: {fn}\n")
            f.write(f"  True Positive: {tp}\n")

        # Plot ROC curve (left side)
        axes[0].plot(fpr, tpr, label=f'{dataset_mode} (AUC = {auc_val:.2f})')

        # Compute Precision-Recall
        precision_curve, recall_curve, _ = precision_recall_curve(y, preds)

        # Plot PR curve (right side)
        axes[1].plot(recall_curve, precision_curve, lw=2, label=f'{dataset_mode} (AUC = {pr_auc:.2f})')

    # Format ROC subplot
    axes[0].plot([0, 1], [0, 1], 'k--')  # Diagonal line
    axes[0].set_xlabel('False Positive Rate', fontsize=18)
    axes[0].set_ylabel('True Positive Rate', fontsize=18)
    axes[0].set_title(f'ROC Curve for {model_name}', fontsize=24)
    axes[0].legend(loc='lower right', fontsize=14)
    axes[0].grid(True)

    # Format Precision-Recall subplot
    axes[1].set_xlabel('Recall', fontsize=18)
    axes[1].set_ylabel('Precision', fontsize=18)
    axes[1].set_title(f'Precision-Recall Curve for {model_name}', fontsize=24)
    axes[1].legend(loc="lower right", fontsize=14)
    axes[1].grid(True)

    # Save the combined figure
    plot_save_path = os.path.join(save_dir, f"{model_name}_ROC_PR_Curves.png")
    plt.savefig(plot_save_path)
    plt.close()  # Close the figure to free memory

    # Cleanup memory
    del model
    K.clear_session()
    gc.collect()

    print(f"Finished all evaluations for {model_name}. Metrics saved in {metrics_file}, and ROC & PR curves saved as {plot_save_path}\n")








