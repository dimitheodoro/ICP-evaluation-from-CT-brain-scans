
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate,Reshape,Embedding, Input,Conv2D, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization,GlobalAveragePooling3D
from keras.regularizers import l1
from keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import Adam,SGD
from tensorflow.keras.initializers import HeNormal
import tensorflow as tf
from tensorflow.keras.losses import BinaryFocalCrossentropy
import numpy as np 
from classification_models_3D.tfkeras import Classifiers


def MultipleInputsDenseNet201(input_shape, sex_label_shape, age_label_shape, GCS_label_shape, 
                              sex_num_classes=2, age_num_classes=3, GCS_num_classes=3):
    Densenet201, _ = Classifiers.get('densenet201')
    
    initializer = HeNormal(seed=42)
    
    # Input for sex label
    sex_label = Input(sex_label_shape)  # shape=(1,)
    sex = Embedding(sex_num_classes, 50)(sex_label)
    print(f"Shape after sex Embedding: {sex.shape}")
    
    n_nodes = input_shape[0] * input_shape[1]
    sex = Dense(n_nodes)(sex)
    print(f"Shape after sex Dense: {sex.shape}")
    
    sex_out = Reshape((input_shape[0], input_shape[1], 1))(sex)
    print(f"Shape after sex Reshape: {sex_out.shape}")
    
    # Input for age label
    age_label = Input(age_label_shape)
    age = Embedding(age_num_classes, 50)(age_label)
    print(f"Shape after age Embedding: {age.shape}")
    
    age = Dense(n_nodes)(age)
    print(f"Shape after age Dense: {age.shape}")
    
    age_out = Reshape((input_shape[0], input_shape[1], 1))(age)
    print(f"Shape after age Reshape: {age_out.shape}")
    
    # Input for GCS label
    GCS_label = Input(GCS_label_shape)
    GCS = Embedding(GCS_num_classes, 50)(GCS_label)
    print(f"Shape after GCS Embedding: {GCS.shape}")
    
    GCS = Dense(n_nodes)(GCS)
    print(f"Shape after GCS Dense: {GCS.shape}")
    
    GCS_out = Reshape((input_shape[0], input_shape[1], 1))(GCS)
    print(f"Shape after GCS Reshape: {GCS_out.shape}")
    
    # Merge the embeddings
    merge_embed = Concatenate()([sex_out, age_out, GCS_out])
    print(f"Shape after Concatenate embeddings: {merge_embed.shape}")
    
    # Input for the image
    input_image = Input(shape=input_shape)
    print(f"Shape of input image: {input_image.shape}")
    
    # Concatenate the image and the embeddings
    merge = Concatenate()([input_image, merge_embed])
    print(f"Shape after Concatenate image and embeddings: {merge.shape}")

    merge = Reshape((merge.shape[1],merge.shape[2],merge.shape[3],1))(merge)
    print(f"Shape after Concatenate image and embeddings 2: {merge.shape}")

    merge = Concatenate()([merge,merge,merge])
    print(f"Shape after Concatenate image and embeddings 3: {merge.shape}")
    

    # Base model - DenseNet201

    base_model = Densenet201(input_shape=(128, 128, 123, 3), weights='imagenet',include_top=False)
    base_model = base_model(merge)
    print(f"Shape after DenseNet201: {base_model.shape}")

    
    # Global average pooling
    x = GlobalAveragePooling3D()(base_model)
    print(f"Shape after GlobalAveragePooling3D: {x.shape}")
    
    # Dense layers
    x = Dense(1024, activation='relu', kernel_regularizer=l1(0.001), kernel_initializer=initializer)(x)
    print(f"Shape after first Dense (1024 units): {x.shape}")
    
    x = Dense(512, activation='relu', kernel_regularizer=l1(0.001), kernel_initializer=initializer)(x)
    print(f"Shape after second Dense (512 units): {x.shape}")
    
    output_layer = Dense(1, activation="sigmoid", kernel_regularizer=l1(0.001), kernel_initializer=initializer)(x)
    print(f"Shape after output Dense layer: {output_layer.shape}")
    
    # Create the model
    model = Model([input_image, sex_label, age_label, GCS_label], outputs=output_layer)
    
    # Compile the model
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=10000,
        decay_rate=0.9
    )
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy'], run_eagerly=True)
    
    return model



if __name__ == '__main__':
    # Step 1: Define the input shapes
    input_shape = (128, 128, 120)
    sex_label_shape = (1,)
    age_label_shape = (1,)
    GCS_label_shape = (1,)

    # Step 2: Generate random data for testing
    input_image = np.random.random((1, *input_shape))  # A single image with the input shape
    sex_label = np.random.randint(0, 2, (1, *sex_label_shape))  # Random sex label (0 or 1)
    age_label = np.random.randint(0, 4, (1, *age_label_shape))  # Random age label (0, 1, or 2)
    GCS_label = np.random.randint(0, 3, (1, *GCS_label_shape))  # Random GCS label (0, 1, or 2)

    # Step 3: Create the model and print the summary
    model = MultipleInputsDenseNet201(input_shape, sex_label_shape, age_label_shape, GCS_label_shape)
    model.summary()
































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







