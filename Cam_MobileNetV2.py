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

os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3,4,5,6,7"
# strategy = tf.distribute.MirroredStrategy()

# patients1 = np.load("/raid/theodoropoulos/PhD/Data/128x128x120/Train/patients_Train.npy",allow_pickle=True)
patients2 = np.load("/raid/theodoropoulos/PhD/Data/128x128x120/Test/patients_Test.npy",allow_pickle=True)

# np.random.seed(42)
np.random.shuffle(patients2)
patients= patients2[-4::]


# patients = np.concatenate((patients1,patients2),axis=0)
# del patients1,patients2

# np.random.seed(42)
# np.random.shuffle(patients)

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


# X_val,  X_test,  y_val, y_test,   labels_sex_val, labels_sex_test,  labels_age_val, labels_age_test ,label_GCS_val, label_GCS_test= train_test_split( X_val, y_val,
#                                                                                                                                                                 labels_sex_val,
#                                                                                                                                                                 labels_age_val,
#                                                                                                                                                                 label_GCS_val,
#                                                                                                                                                                 test_size=0.1,random_state=42)

print(X_val.shape)
batch_size = 1
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



# print("loaded dataset")
# Paths to the saved models
model_paths = {
#   "Custom 3D": "/raid/theodoropoulos/PhD/Results/whole image/128_128_120/3D/3D_full_image_model_120_128_128_Tensorflow_16.keras",
#    "DenseNet201": '/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/DenseNet201_120_128_128.keras',
#     "ResNet101": '/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/Resnet101_120_128_128.keras',
 "MobileNetV2": '/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/MobileNetV2_120_128_128.keras',
#   "Seresnet18": '/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/Seresnet18_120_128_128.keras'
}


# print("loaded paths")
# Dictionary to store the results
# results = []

# Plot ROC curves for all models
# plt.figure(figsize=(10, 8))

# del patients
Test_image = np.squeeze(X_val)

for model_name, model_path in model_paths.items():
    # Load the model
    model = tf.keras.models.load_model(model_path,compile=False)
    print("Loading....:{}".format(model_name))
    ##########################################


print(model.summary())
# ###############################################################################################

last_layer_weights = model.layers[-1].get_weights()[0]  #Predictions layer
print("last_layer_weights shape",last_layer_weights.shape)
# pred_vec = model.predict(validation_dataset)
# print("pred_vec",pred_vec)


# print("last_layer_weights 0 shape ",last_layer_weights[:,0].shape)
# print("last_layer_weights 1 shape ",last_layer_weights[:,1].shape)

# if pred_vec<0.5:
#     pred_class_=0
# elif pred_vec>0.5:
#     pred_class_=1

# pred_class = np.argmax(pred_vec)




last_layer_weights_for_pred = last_layer_weights # dim: (512,) 
# print(last_layer_weights_for_pred.shape)  


CAM_model = Model(inputs=model.input, outputs=model.layers[-4].input)
last_conv_output = CAM_model.predict(validation_dataset)
print(last_conv_output.shape) #(1, 4, 4, 3, 1920)
last_conv_output = np.squeeze(last_conv_output) #7x7x2048
print(last_conv_output.shape) #( 4, 4, 3, 1920)

# spline interpolation to resize each filtered image to size of original image 
import scipy

h = float(128/last_conv_output.shape[0])
w = float(128/last_conv_output.shape[1])
d = float(123/last_conv_output.shape[2])
z = float(512/last_conv_output.shape[3])
upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, d ,z), order=1) 
print(upsampled_last_conv_output.shape) # (128, 128, 123, 512)

# Reshape back to the image size for easy overlay onto the original image. 

heat_map = np.dot(upsampled_last_conv_output.reshape((-1,512)), ############################################## to 1280 άλλαξα!!
                last_layer_weights_for_pred).reshape(128,128,123)

# heat_map = np.dot(upsampled_last_conv_output.reshape((224*224, 2048)), 
#                   last_layer_weights_for_pred).reshape(224,224) # dim: 224 x 224

print("Done..")

print("heat map shape: ",heat_map.shape)


plt.figure(figsize=(10,10))
# plt.imshow(Test_image[:,:,22].astype('float32'),cmap='gray')
# plt.imshow(heat_map[:,:,22], cmap='jet', alpha=0.50)
# plt.savefig("heatmap.png")



for i in range(4):
    SLICE = np.random.randint(1, 121)
    plt.subplot(2,2,i+1),plt.imshow(Test_image[:,:,SLICE].astype('float32'),cmap='gray')
    plt.title("slide: {}".format(SLICE))
    plt.imshow(heat_map[:,:,SLICE], cmap='jet', alpha=0.4)
    plt.axis('off')  # Turn off the axis

if y_val==0:
    NORMALITY='Normal'
else:
    NORMALITY='Pathological'

plt.suptitle("Heat map of a {} patient with pretrained MobileNetV2".format(NORMALITY),fontsize=16)
plt.savefig("heatmap_MobileNetV2.png")
#######################################################################################























