from skimage import morphology
import pydicom
import numpy as np
from scipy import ndimage

def segment(brain_image):
    segmentation = morphology.dilation(brain_image, np.ones((1,1)))
    labels, label_nb = ndimage.label(segmentation)
    label_count = np.bincount(labels.ravel().astype('int'))
    # The size of label_count is the number of classes/segmentations found
    
    # We don't use the first class since it's the background
    label_count[0] = 0
    
    # We create a mask with the class with more pixels
    # In this case should be the brain
    mask = labels == label_count.argmax()

    # Improve the brain mask
    mask = morphology.dilation(mask, np.ones((5, 5)))
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))
    # masked_image = mask * brain_image

    return mask

def segment_all_patients_slices(X):

    def find_threshold(X):
        counts, bins = np.histogram(X, bins=10)
        bin_max_index =np.argmax(counts)
        # volume[volume<=bins[bin_max_index+1]]=0
        return bins[bin_max_index+1]
    
    PATIETNTS_THRESHOLDS=[[] for _ in range(X.shape[0])]
    for patient in range(X.shape[0]):
        for slice in range(X.shape[-1]):
            PATIETNTS_THRESHOLDS[patient].append(find_threshold(X[patient,:,:,slice]))
  
    X_segmented = []
    for patient in range(X.shape[0]):
        for slice in range(X.shape[-1]):
            image = X[patient][:,:,slice] #########################
            thresholded = image>PATIETNTS_THRESHOLDS[patient][slice]
            mask = segment(thresholded).astype('float64')
            X_segmented.append(image*mask)

    X_segmented = np.array(X_segmented)
    X_segmented = np.reshape(X_segmented,(X.shape[0],X.shape[-1],X.shape[1],-1))
    X_segmented = np.transpose(X_segmented,(0,2,3,1))
    # print(thresholded.dtype,mask.dtype,X_segmented.dtype)
    return X_segmented
