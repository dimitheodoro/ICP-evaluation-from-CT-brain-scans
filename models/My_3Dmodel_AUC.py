
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate,Reshape,Embedding, Input,Conv2D, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization,GlobalAveragePooling2D
from keras.regularizers import l1
from keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import Adam,SGD
from tensorflow.keras.initializers import HeNormal
import tensorflow as tf
from tensorflow.keras.losses import BinaryFocalCrossentropy
import numpy as np 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate,Reshape,Embedding, Input,Conv3D, MaxPooling3D, Flatten, Dense, Dropout,BatchNormalization,GlobalAveragePooling3D
from keras.regularizers import l1
from keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import Adam,SGD
from tensorflow.keras.initializers import HeNormal


def MultipleInputsModel_3D(input_shape, sex_label_shape, age_label_shape,GCS_label_shape, sex_num_classes=2, age_num_classes=3,GCS_num_classes=3):
    neg = 394
    pos = 125
    output_bias = tf.keras.initializers.Constant(np.log([pos/neg]))
    initializer = HeNormal(seed=42)
    # Functional API
    sex_label =Input(sex_label_shape)  #shape=(1,)
    sex = Embedding (sex_num_classes,50)(sex_label)
    n_nodes = input_shape[0] * input_shape[1] 
    sex = Dense(n_nodes)(sex)
    sex_out = Reshape((input_shape[0],input_shape[1],1))(sex)

    age_label =Input(age_label_shape)
    age = Embedding (age_num_classes,50)(age_label)
    age = Dense(n_nodes)(age)
    age_out = Reshape((input_shape[0],input_shape[1],1))(age)

    GCS_label =Input(GCS_label_shape)
    GCS = Embedding (GCS_num_classes,50)(GCS_label)
    GCS = Dense(n_nodes)(GCS)
    GCS_out = Reshape((input_shape[0],input_shape[1],1))(GCS)


    merge_embed = Concatenate()([sex_out,age_out,GCS_out])
    input_image = Input(shape=input_shape)
    merge = Concatenate()([input_image,merge_embed])
    input_3d = Reshape((128,128,123,1))(merge)

    x = Conv3D(64, kernel_size=3, activation='relu',kernel_initializer=initializer, padding='same')(input_3d)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x) 

    x = Conv3D(64, kernel_size=3, activation='relu',kernel_initializer=initializer, padding='same')(input_3d)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x) 

    x = Conv3D(128,kernel_size=3, activation='relu',kernel_initializer=initializer, padding='same')(x)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x) 

    x = Conv3D(256,kernel_size=3, activation='relu',kernel_initializer=initializer, padding='same')(x)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x) 
        
    x = GlobalAveragePooling3D()(x)
    
    #x = Dense(1024, activation='relu', kernel_regularizer=l1(0.001),kernel_initializer=initializer)(x)
    #x = Dense(512, activation='relu', kernel_regularizer=l1(0.001),kernel_initializer=initializer)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l1(0.001),kernel_initializer=initializer)(x) ############################################
    x = Dropout(0.3)(x) 
    
    output_layer = Dense(1, activation="sigmoid", kernel_regularizer=l1(0.001),kernel_initializer=initializer,bias_initializer=output_bias)(x)
    # Create the model
    model = Model([input_image,sex_label,age_label,GCS_label], outputs=output_layer)
    # Compile the model
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'],run_eagerly=True)
    lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
              metrics=[tf.keras.metrics.AUC(name='auc'),'accuracy'],run_eagerly=True)
    # model.compile(optimizer=optimizer, loss=BinaryFocalCrossentropy(apply_class_balancing=True,from_logits=False),
    #          metrics=[tf.keras.metrics.Recall()],run_eagerly=True)
    # model.compile(optimizer=optimizer, loss='binary_crossentropy',
    #           metrics=['accuracy', tf.keras.metrics.F1Score(threshold=0.5)],run_eagerly=True)
    # model.compile(optimizer=optimizer, loss='binary_crossentropy',
    #           metrics=[ tf.keras.metrics.AUC()],run_eagerly=True)
    # model.compile(optimizer=optimizer, loss=BinaryFocalCrossentropy(apply_class_balancing=True,from_logits=False),
    #           metrics=[ tf.keras.metrics.AUC()],run_eagerly=True)
    # model.compile(optimizer=optimizer, loss=BinaryFocalCrossentropy(apply_class_balancing=True,from_logits=False,alpha=0.75),
    #           metrics=[tf.keras.metrics.Recall()],run_eagerly=True)
#    model.compile(optimizer=optimizer, loss='binary_crossentropy',
#              metrics=[tf.keras.metrics.TruePositives(), tf.keras.metrics.FalsePositives(),
#                       tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalseNegatives(),tf.keras.metrics.AUC(name="AUC")],run_eagerly=True)
   # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'],run_eagerly=True)
    return model






































def CNN_TURBO(input_shape):
    initializer = HeNormal(seed=42)
    # Functional API
    input_image = Input(shape=input_shape)
    
    x = Conv2D(64, (3, 3), activation='relu',kernel_initializer=initializer)(input_image)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x) 

    x = Conv2D(64, (3, 3), activation='relu',kernel_initializer=initializer)(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x) 

    x = Conv2D(128, (3, 3), activation='relu',kernel_initializer=initializer)(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x) 

    x = Conv2D(256, (3, 3), activation='relu',kernel_initializer=initializer)(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x) 

    x = Conv2D(256, (3, 3), activation='relu',kernel_initializer=initializer)(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x) 
    
    x = Flatten()(x)
    
    x = Dense(1024, activation='relu', kernel_regularizer=l1(0.001),kernel_initializer=initializer)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l1(0.001),kernel_initializer=initializer)(x)
    output_layer = Dense(1, activation="sigmoid", kernel_regularizer=l1(0.001),kernel_initializer=initializer)(x)
    # Create the model
    model = Model(inputs=input_image, outputs=output_layer)
    # Compile the model
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'],run_eagerly=True)
    lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
    optimizer = Adam(learning_rate=lr_schedule)
    #model.compile(optimizer=optimizer, loss='binary_crossentropy',
    #         metrics=['accuracy', tf.keras.metrics.Recall()],run_eagerly=True)
    # model.compile(optimizer=optimizer, loss='binary_crossentropy',
    #           metrics=[ tf.keras.metrics.F1Score(threshold=0.5)],run_eagerly=True)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
              metrics=[ tf.keras.metrics.AUC()],run_eagerly=True)
    #model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'],run_eagerly=True)
    return model







