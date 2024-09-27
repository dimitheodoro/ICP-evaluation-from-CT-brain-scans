import tensorflow as tf
import numpy as np
from scipy import ndimage
import random
import cv2
from scipy.ndimage import gaussian_filter

@tf.function
def CT_augmentations(volume,y):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        
        if tf.reduce_all(tf.equal(volume, 0)).numpy():
            return volume.astype('float64')
        else:
             # Define some rotation angles
            angles = [-20, -10, -5, 5, 10, 20]
            # Pick an angle at random
            angle = random.choice(angles)
            # Rotate volume
            volume = ndimage.rotate(volume, angle, reshape=False)
            volume[volume < 0] = 0
            volume[volume > 1] = 1
            # print(volume)
            # print(volume)
            return volume.astype('float64')
            
    
    def gaussianfilter(volume):
        if tf.reduce_all(tf.equal(volume, 0)).numpy():
            return volume.astype('float64')
        else:
            # print(gaussian_filter(volume, sigma=1))
            return gaussian_filter(volume, sigma=1).astype('float64')
            

    # def apply_augmentation(volume,y):
    #     volume_augmented = volume
    #     if y==1:
    #         # print("class {} scipy_rotate".format(y))
    #         volume_augmented = scipy_rotate(volume_augmented)
    #     if y==1:
    #         # print("class {} gaussianfilter".format(y))
    #         volume_augmented = gaussianfilter(volume_augmented)
    #     if y==0:pass 
    #         # print("class {} pass...".format(y))
    #     return volume_augmented

    def apply_augmentation(volume,y):
        if y==1 and random.random() < 0.5:
            volume_augmented_scipy = scipy_rotate(volume)
            return volume_augmented_scipy
        elif y==1 and random.random() > 0.5:
            volume_augmented_gauss = gaussianfilter(volume)
            return volume_augmented_gauss
        else:
            return volume
    

    def min_max_normalize(array):
        """
        Normalize a SymbolicTensor to the range [0, 1] using min-max normalization.
        
        Parameters:
        array (SymbolicTensor): The input SymbolicTensor to be normalized.
        
        Returns:
        SymbolicTensor: The normalized SymbolicTensor.
        """
        # Convert SymbolicTensor to a TensorFlow tensor
        array_tf = tf.convert_to_tensor(array)
        
        # Calculate min and max values
        min_val = tf.reduce_min(array_tf)
        max_val = tf.reduce_max(array_tf)

        if min_val==max_val:
            max_val += 1e-10
            # Apply min-max normalization
            normalized_array = (array_tf - min_val) / (max_val - min_val)
            return normalized_array
        elif min_val!=max_val:
            normalized_array = (array_tf - min_val) / (max_val - min_val)
            return normalized_array
    

    volume = tf.numpy_function(apply_augmentation,[volume,y], 'float64','int32')
    volume = tf.numpy_function(min_max_normalize, [volume], 'float64')
    # volume = tf.numpy_function(set_zero_to_image, [volume], 'float64')
    return volume
      