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
from models import MultipleInputs3DMobileNetV2, MultipleInputs3DResNet101, MultipleInputs3DDenseNet201
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay

# Other imports and augmentations remain the same

SAVE = True
epochs = 50

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
strategy = tf.distribute.MirroredStrategy()
patients1 = np.load("/path/to/data/patients_Train.npy", allow_pickle=True)
patients2 = np.load("/path/to/data/patients_Test.npy", allow_pickle=True)

patients = np.concatenate((patients1, patients2), axis=0)
np.random.seed(42)
np.random.shuffle(patients)

X = np.array([patients[i]['volume'] for i in range(len(patients))])
X = np.transpose(X, (0, 2, 3, 1))
X = segment_all_patients_slices(X)
print(X.shape)
y = np.array([patients[i]['Class'] for i in range(len(patients))]).astype('int32')

# Label encoding and preprocessing remain the same

X_train, X_val, y_train, y_val, labels_sex_train, labels_sex_val, labels_age_train, labels_age_val, label_GCS_train, label_GCS_val = train_test_split(
    X, y, labels_sex_transf, labels_age_transf, labels_GCS_transf, test_size=0.3, random_state=42)

X_val, X_test, y_val, y_test, labels_sex_val, labels_sex_test, labels_age_val, labels_age_test, label_GCS_val, label_GCS_test = train_test_split(
    X_val, y_val, labels_sex_val, labels_age_val, label_GCS_val, test_size=0.3, random_state=42)

batch_size = 8

train_loader = tf.data.Dataset.from_tensor_slices((X_train, labels_sex_train, labels_age_train, label_GCS_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((X_val, labels_sex_val, labels_age_val, label_GCS_val, y_val))
test_loader = tf.data.Dataset.from_tensor_slices((X_test, labels_sex_test, labels_age_test, label_GCS_test, y_test))

train_dataset = train_loader.map(train_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
validation_dataset = validation_loader.map(validation_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_loader.map(test_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
class_weight = {0: class_weight[0], 1: class_weight[1]}
print(class_weight)

model_constructors = {
    "3D MobileNetV2": MultipleInputs3DMobileNetV2,
    "3D ResNet101": MultipleInputs3DResNet101,
    "3D DenseNet201": MultipleInputs3DDenseNet201
}

for model_name, model_constructor in model_constructors.items():
    print(f"Training {model_name} model...")
    model = model_constructor(input_shape=(128, 128, 120), sex_label_shape=(1,), age_label_shape=(1,), GCS_label_shape=(1,),
                              age_num_classes=len(np.unique(labels_age_transf)),
                              sex_num_classes=len(np.unique(labels_sex_transf)),
                              GCS_num_classes=len(np.unique(labels_GCS_transf)))

    with strategy.scope():
        distributed_model = model

    for layer in distributed_model.layers[17:18]:
        layer.trainable = False

    lr_schedule = ExponentialDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)
    optimizer = Adam(learning_rate=lr_schedule)
    distributed_model.compile(optimizer=optimizer, loss='binary_crossentropy',
                              metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy'], run_eagerly=True)

    history = distributed_model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, shuffle=True, verbose=2, class_weight=class_weight)

    print("calculating AUC...")
    preds = distributed_model.predict(validation_dataset)
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

    j_scores = tpr - fpr
    optimal_idx_roc = np.argmax(j_scores)
    optimal_threshold_roc = thresholds[optimal_idx_roc]

    threshold = optimal_threshold_roc
    y_pred = (preds > threshold).astype('float')
    tn, fp, fn, tp = confusion_matrix(y_true=y_val, y_pred=y_pred).ravel()
    print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")

    f1 = f1_score(y_true=y_val, y_pred=y_pred)
    print(f"f1-score: {f1}")

    recall = tp / (tp + fn)
    print(f"Recall: {recall}")

    precision = tp / (tp + fp)
    print(f"Precision: {precision}")

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print(f"Accuracy: {accuracy}")

    if SAVE:
        plt.savefig(f"/path/to/save/{model_name}_loss_acc_AUC(120,128,128).png")
        distributed_model.save(f"/path/to/save/{model_name}_120_128_128.keras")
