
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA

def midline(all_sclices_as_bool_mask,name,graph=True):
    mask = np.where(all_sclices_as_bool_mask, 255, 0)

    # Find the coordinates of the white pixels
    x, y = np.where(mask == 255)

    # Calculate the centroid of the white pixels
    cx = np.mean(x)
    cy = np.mean(y)

    # Perform principal component analysis on the white pixels
    pca = PCA(n_components=2)
    pca.fit(np.column_stack((x, y)))

    # Get the direction of the principal axis
    direction = pca.components_[0]
    # Calculate the angle between the principal axis and the y-axis
    angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi  #-170
    angle_perpendicular = 180 -angle   #344.95 
    
    if angle_perpendicular> 180:
        DIRECTION = 'LEFT'
    else:
        DIRECTION ='RIGHT'

    if DIRECTION=='RIGHT':
        angle_perpendicular = 180 - np.abs(angle)
    else:
        angle_perpendicular = angle_perpendicular-360

    
    # print("Angle of the midline: {:.2f} degrees".format(angle))
    # print("Angle of the y-axis (perpendicular): {:.2f} degrees".format(angle_perpendicular))

    if graph:
        # Plot the image, the centroid, and the principal axis

        plt.imshow(mask, cmap='gray')
        plt.scatter(cy, cx, c='r', marker='+')
        plt.plot([cy, cy + direction[1] * 100], [cx, cx + direction[0] * 100], c='r')
        plt.axis('off')
        plt.title('{},angle:{:.2f}'.format(name,angle_perpendicular),fontsize=7)
        plt.show()
    return angle_perpendicular