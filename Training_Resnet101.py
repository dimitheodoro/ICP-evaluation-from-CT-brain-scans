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
from My_model_Resnet101 import MultipleInputsResnet101
# from CT_DATASET_module_with_Classes_rescale import *
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage
import random
from augmentationsMINORITY import CT_augmentations
from sklearn.utils.class_weight import compute_class_weight
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve,auc
#from tensorflow.keras.backend import clear_session
#clear_session()
from classification_models_3D.tfkeras import Classifiers
from keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import Adam,SGD


SAVE=True
epochs = 50

os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3,4,5,6,7"
strategy = tf.distribute.MirroredStrategy()
patients1 = np.load("/raid/theodoropoulos/PhD/Data/128x128x120/Train/patients_Train.npy",allow_pickle=True)
patients2 = np.load("/raid/theodoropoulos/PhD/Data/128x128x120/Test/patients_Test.npy",allow_pickle=True)

patients = np.concatenate((patients1,patients2),axis=0)
# Set the seed for reproducibility
np.random.seed(42)
np.random.shuffle(patients)
#patients=patients[:30]

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




model =MultipleInputsResnet101(input_shape=(128,128,120),sex_label_shape=(1,),age_label_shape=(1,),GCS_label_shape=(1,),
                           age_num_classes=len(np.unique(labels_age_transf)),
                           sex_num_classes=len(np.unique(labels_sex_transf)),
                           GCS_num_classes=len(np.unique(labels_GCS_transf)),
                           )
####################
################################################################## CT augmetations #########################

import numpy as np
import random
from scipy.ndimage import rotate, gaussian_filter
import tensorflow as tf

def scipy_rotate(volume):
    if tf.reduce_all(tf.equal(volume, 0)).numpy():
        return volume.astype('float64')
    else:
        angles = [-20, -10, -5, 5, 10, 20]
        angle = random.choice(angles)
        volume = rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume.astype('float64')

def gaussianfilter(volume):
    if tf.reduce_all(tf.equal(volume, 0)).numpy():
        return volume.astype('float64')
    else:
        return gaussian_filter(volume, sigma=1).astype('float64')

@tf.function
def CT_augmentations(volume, y):
    """Apply augmentation to CT images based on their labels."""
    def py_augment(volume, y):
        if y == 1 and random.random() < 0.5:
            volume_augmented = scipy_rotate(volume)
        elif y == 1 and random.random() >= 0.5:
            volume_augmented = gaussianfilter(volume)
        else:
            volume_augmented = volume
        return volume_augmented

    volume_augmented = tf.numpy_function(py_augment, [volume, y], tf.float64)
    return volume_augmented

def train_preprocessing(volume, labels_sex, labels_age, labels_gcs, y):
    volume = CT_augmentations(volume, y)
    volume = tf.expand_dims(volume, axis=3)
    return (volume, labels_sex, labels_age, labels_gcs), y

def validation_preprocessing(volume, labels_sex, labels_age, labels_gcs, y):

    volume = tf.expand_dims(volume, axis=3)
    return (volume, labels_sex, labels_age, labels_gcs), y

def test_preprocessing(volume, labels_sex, labels_age, labels_gcs, y):

    volume = tf.expand_dims(volume, axis=3)
    return (volume, labels_sex, labels_age, labels_gcs), y

# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((X_train,    
                                                     labels_sex_train, 
                                                     labels_age_train, 
                                                     label_GCS_train, 
                                                     y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((X_val,    
                                                     labels_sex_val, 
                                                     labels_age_val, 
                                                     label_GCS_val, 
                                                     y_val))

test_loader = tf.data.Dataset.from_tensor_slices((X_test,    
                                                     labels_sex_test, 
                                                     labels_age_test, 
                                                     label_GCS_test, 
                                                     y_test))

batch_size = 8

train_dataset = (
    #train_loader.shuffle(len(X_train))
    train_loader.map(train_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

validation_dataset = (
    #validation_loader.shuffle(len(X_val))
    validation_loader.map(validation_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

test_dataset = (
    #test_loader.shuffle(len(X_test))
    test_loader.map(test_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)


########################################

class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
class_weight = {0:class_weight[0], 1:class_weight[1]}
print(class_weight)
####################
# Create and compile the distributed model
with strategy.scope():
	distributed_model = model



# # ##################################################### Train for some epochs ###################################

# for i,layer in enumerate(distributed_model.layers[17:18]):

#     layer.trainable = False #All others as non-trainable

# # Compile the model
# distributed_model.compile(optimizer=SGD, loss='binary_crossentropy',
#                 metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy'], run_eagerly=True)

# distributed_model.fit(
# train_dataset,
# validation_data=validation_dataset,
# epochs=3,
# shuffle=True,
# verbose=2,
# class_weight=class_weight,
# # callbacks=[checkpoint_cb, early_stopping_cb],
# )

################################################# train normally 
for i,layer in enumerate(distributed_model.layers[17:18]):
  
    layer.trainable = False #All others as non-trainable


# Define the EarlyStopping callback
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Compile the model
lr_schedule = ExponentialDecay(
    initial_learning_rate=0.05,
    decay_steps=100,
    decay_rate=0.45
)
optimizer = Adam(learning_rate=lr_schedule)
distributed_model.compile(optimizer=optimizer, loss='binary_crossentropy',
                metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy'], run_eagerly=True)



history = distributed_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    class_weight=class_weight,
    # callbacks=[checkpoint_cb, early_stopping_cb],
)

###############################################################################################################
print("calculating AUC...")
preds = distributed_model.predict(validation_dataset)
#preds = preds[:len(y_val)]  # Adjust the length of preds to match y_val if necessary
# Calculate AUC
fpr, tpr, thresholds = roc_curve(y_val, preds)
AUC = auc(fpr, tpr)
print("AUC: {:.3f}".format(AUC))
print("done!")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history['auc'], label='Training AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
plt.title('Training and Validation AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')

plt.tight_layout()
plt.show()

##############################################################

# Compute Youden's J statistic for each threshold
j_scores = tpr - fpr
optimal_idx_roc = np.argmax(j_scores)
optimal_threshold_roc = thresholds[optimal_idx_roc]


# confusion matrix
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
threshold = optimal_threshold_roc
y_pred = (preds > threshold).astype('float')
tn, fp, fn, tp = confusion_matrix(y_true=y_val, y_pred=y_pred).ravel()
print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")

# f1 score
f1 = f1_score(y_true=y_val, y_pred=y_pred) # 0.875 high
f1_test = 2*tp/(2*tp + fn + fp)
print(f"f1-score: {f1}")

# Recall
recall = tp / (tp + fn)
print(f"Recall: {recall}")

# Precision
precision = tp / (tp + fp)
print(f"Precision: {precision}")

# Accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f"Accuracy: {accuracy}")

##############################################################

if SAVE:
    plt.savefig("/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/Resnet101_loss_acc_AUC(120,128,128).png")
    distributed_model.save("/raid/theodoropoulos/PhD/Results/whole image/128_128_120/AUC/Resnet101_120_128_128.keras")
