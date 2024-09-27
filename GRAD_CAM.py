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
# from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score,recall_score,precision_score,precision_recall_curve, accuracy_score,roc_auc_score
from keras.models import Model 
import scipy

os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2"
# strategy = tf.distribute.MirroredStrategy()

# patients1 = np.load("/raid/theodoropoulos/PhD/Data/128x128x120/Train/patients_Train.npy",allow_pickle=True)
patients2 = np.load("/raid/theodoropoulos/PhD/Data/128x128x120/Test/patients_Test.npy",allow_pickle=True)

# RANDOMNESS = 'None' 

# RANDOMNESS_SLIDES = 25
#np.random.seed(RANDOMNESS)
#np.random.shuffle(patients2)
#patients= patients2[-4::]
PATIENTS =11,13,23
Test_image = []
patient_predicted_all = []

model_paths = {
#   "Custom 3D": "/raid/theodoropoulos/PhD/Results/whole image/128_128_120/3D/3D_full_image_model_120_128_128_Tensorflow_16.keras",
# "DenseNet201": '/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/DenseNet201_120_128_128.keras',
    # "ResNet101": '/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/Resnet101_120_128_128.keras',
"MobileNetV2": '/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/MobileNetV2_120_128_128.keras',
#   "Seresnet18": '/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/Seresnet18_120_128_128.keras'
}
# Load the model
model = tf.keras.models.load_model(model_paths['MobileNetV2'],compile=False)
print("Loading....:{}".format(model_paths['MobileNetV2'].split('/')[-1].split('_')[0]))

##########################################

def make_gradcam_heatmap(img_array, model, pred_index=0):
    global class_channel
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model(
        model.inputs, [model.layers[-4].input, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    print('last_conv_layer_output shape:',last_conv_layer_output.shape)
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))
    print('pooled_grads shape:',pooled_grads.shape)
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


for PATIENT in PATIENTS:
    print("Patient:",PATIENT)
    patients = patients2[PATIENT],patients2[PATIENT],patients2[PATIENT],patients2[PATIENT]

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
                                                                                                                                                                    test_size=0.1,random_state=42)

   
    batch_size = 1


    Test_image.append(np.squeeze(X_val))

    MODELS=[]
    HEAT_MAPS=[]

    for model_name, _ in model_paths.items():

        heatmap = make_gradcam_heatmap(img_array=(X_val,    
                                                        labels_sex_val, 
                                                        labels_age_val, 
                                                        label_GCS_val, ),model=model)
        h = 128
        w = 128
        d = 123
        
        if class_channel>0.5 and y_val==1:
            patient_predicted='Patient correctly predicted \n      as pathological'
        elif class_channel<0.5 and y_val==0:
            patient_predicted='Patient correctly predicted \n            as normal'
        elif class_channel<0.5 and y_val==1:
            patient_predicted='Patient wrongly predicted \n              as normal'
        elif class_channel>0.5 and y_val==0: 
            patient_predicted='Patient wrongly predicted \n        as pathological'

        patient_predicted_all.append(patient_predicted)

        heatmap_resized = scipy.ndimage.zoom(heatmap, (h / heatmap.shape[0], w / heatmap.shape[1], d / heatmap.shape[2] ), order=1) 
        print(heatmap_resized.shape)
        MODELS.append(model_name)
        HEAT_MAPS.append(heatmap_resized)




random_slices = [40,55,75,90]





import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set the figure size
fig = plt.figure(figsize=(20, 12))

# Create a grid with extra rows for gaps
gs = gridspec.GridSpec(9, 4, height_ratios=[1.2, 1.2, 0.2, 1.2, 1.2, 0.2, 1.2, 1.2, 0.2])  # Added height ratio for gap rows

random_slices = [40, 55, 75, 90]

# First pair of rows (patient 1)
for i in range(4):
    SLICE = random_slices[i]
    ax = fig.add_subplot(gs[0, i])  # First row
    ax.imshow(Test_image[0][:, :, SLICE].astype('float32'), cmap='gray')
    ax.axis('off')

for i in range(4):
    SLICE = random_slices[i]
    ax = fig.add_subplot(gs[1, i])  # Second row
    ax.imshow(Test_image[0][:, :, SLICE].astype('float32'), cmap='gray')
    ax.imshow(HEAT_MAPS[0][:, :, SLICE], cmap='jet', alpha=0.4)
    ax.axis('off')

# Add gap (this is the empty third row, so no need to plot anything here)

# Second pair of rows (patient 2)
for i in range(4):
    SLICE = random_slices[i]
    ax = fig.add_subplot(gs[3, i])  # Fourth row (skipped the third row)
    ax.imshow(Test_image[1][:, :, SLICE].astype('float32'), cmap='gray')
    ax.axis('off')

for i in range(4):
    SLICE = random_slices[i]
    ax = fig.add_subplot(gs[4, i])  # Fifth row
    ax.imshow(Test_image[1][:, :, SLICE].astype('float32'), cmap='gray')
    ax.imshow(HEAT_MAPS[0][:, :, SLICE], cmap='jet', alpha=0.4)
    ax.axis('off')

# Add gap (this is the empty sixth row)

# Third pair of rows (patient 3)
for i in range(4):
    SLICE = random_slices[i]
    ax = fig.add_subplot(gs[6, i])  # Seventh row
    ax.imshow(Test_image[2][:, :, SLICE].astype('float32'), cmap='gray')
    ax.axis('off')

for i in range(4):
    SLICE = random_slices[i]
    ax = fig.add_subplot(gs[7, i])  # Eighth row
    ax.imshow(Test_image[2][:, :, SLICE].astype('float32'), cmap='gray')
    ax.imshow(HEAT_MAPS[0][:, :, SLICE], cmap='jet', alpha=0.4)
    ax.axis('off')

# Add gap (this is the empty ninth row)

# Text for the left labels (patient names or predictions)
fig.text(0.05, 0.83, patient_predicted_all[0], fontsize=9, va='center', rotation='horizontal', weight='bold')
fig.text(0.05, 0.56, patient_predicted_all[1], fontsize=9, va='center', rotation='horizontal', weight='bold')
fig.text(0.05, 0.30, patient_predicted_all[2], fontsize=9, va='center', rotation='horizontal', weight='bold')

# Add a main title for the comparison
plt.suptitle("Predictions of MobileNetV2", fontsize=20)

# Save the figure
plt.savefig("GRAD_CAM/XXX_GRAD_CAM_heatmap_of_many_patients_with_gaps.png")

# Show the figure (optional)
plt.show()























