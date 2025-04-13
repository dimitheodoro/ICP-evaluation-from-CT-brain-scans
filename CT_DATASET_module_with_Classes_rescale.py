import pydicom as dcm
import os
import numpy as np
from scipy import ndimage
import re



class CT_DATASET:
    def __init__(self,path,resamling_dimensions):
        self.path = path
        self.resamling_dimensions = resamling_dimensions

    def _sort_dicoms(self,path_of_dicom_slices_from_one_patient):
        acq_times = ([dcm.read_file(os.path.join(path_of_dicom_slices_from_one_patient,slice))[('0008','0032')].value for slice in (os.listdir((path_of_dicom_slices_from_one_patient)))])
        scans_times = dict(zip(os.listdir(path_of_dicom_slices_from_one_patient),acq_times))
        scans_times = {k: v for k, v in sorted(scans_times.items(), key=lambda item: item[1])}
        sorted_dcm = list(scans_times.keys())
        return sorted_dcm
    
    def _window_image(self,img, window_center,window_width, intercept, slope, rescale=True):
        img[img <= -2000] = 0
        img = (img*slope +intercept) #for translation adjustments given in the dicom file. 
        img_min = window_center - window_width//2 #minimum HU level
        img_max = window_center + window_width//2 #maximum HU level
        img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
        img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
        if rescale: 
            img = (img - img_min) / (img_max - img_min)
        return img
    
    def _get_first_of_dicom_field_as_int(self,x):
        #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
        if type(x) == dcm.multival.MultiValue: return int(x[0])
        else: return int(x)

    def _get_windowing(self,data):
        dicom_fields = [data[('0028','1050')].value, #window center
                        data[('0028','1051')].value, #window width
                        data[('0028','1052')].value, #intercept
                        data[('0028','1053')].value] #slope
        return [self._get_first_of_dicom_field_as_int(x) for x in dicom_fields]
    
    
    # def _get_windowing(self,data):

    #     if data[('0008','103e')].value=='ThinSliceSpi  4.0  H31s':
            
    #         dicom_fields = [44, #window center
    #                         78, #window width
    #                         data[('0028','1052')].value, #intercept
    #                         data[('0028','1053')].value] #slope
    #     else:                  
    #         dicom_fields = [data[('0028','1050')].value, #window center
    #                         data[('0028','1051')].value, #window width
    #                         data[('0028','1052')].value, #intercept
    #                         data[('0028','1053')].value] #slope
    #     return [self._get_first_of_dicom_field_as_int(x) for x in dicom_fields]
    
    
    
    def _read_dicom_images(self,path_of_dicom_slices_from_one_patient,folder_of_patient_name):
        global SERIES
        global GCS
        global CLASS
        global window_center, window_width

        GCS =folder_of_patient_name.split(r"^")[-1]
        CLASS = folder_of_patient_name.split(r"^")[0][-1]
        scans = [dcm.read_file(os.path.join(path_of_dicom_slices_from_one_patient, slice)) for slice in self._sort_dicoms(path_of_dicom_slices_from_one_patient)]
        slices = np.array([dcm.read_file(os.path.join(path_of_dicom_slices_from_one_patient, slice)).pixel_array for slice in self._sort_dicoms(path_of_dicom_slices_from_one_patient)])
        # print("The type of ct scan is",scans[0][('0008','103e')].value)
        SERIES = scans[0][('0008','103e')].value 
        ################################   special cases ####################
        # if scans[0][('0008','103e')].value=='ThinSliceSpi  4.0  H31s':
        if scans[0][('0008','103e')].value=='head':
            slices = slices[::2]

        if scans[0][('0008','103e')].value=='ThinSliceSpi  4.0  H31s':pass
  
        # if scans[0][('0008','103e')].value=='Neck  5.0  B31s':
        ################################################################################
        window_center, window_width, intercept, slope = self._get_windowing(scans[0])
        windowed_slices = self._window_image(slices, window_center, window_width, intercept, slope)
        name, sex, age = scans[0][('0010','0010')].value, scans[0][('0010','0040')].value,scans[0][('0010','1010')].value
        # print("SEX FROM DICOM:",sex)
        print(name.original_string)
        TZOURAS = name.original_string
        if sex != "M" and sex != "F":
            if name.original_string==TZOURAS:
                sex = "M"
            elif str(name.original_string).split('b')[-1].split("^")[1][-1] == 'S':
                sex = "M"
            elif str(name.original_string).split('b')[-1].split("^")[1][-1] == 'L':
                sex = "M"
            elif str(name.original_string).split('b')[-1].split("^")[1][-1] == 'N':
                sex = "M"
            elif str(name.original_string).split('b')[-1].split("^")[0][1:8] == 'AGNOSTO' and str(name.original_string).split('b')[-1].split("^")[1][-2] == 'S':
                sex = "M"
            elif str(name.original_string).split('b')[-1].split("^")[0][1:8] == 'AGNOSTO':
                sex = "NA"
            elif str(name.original_string).split('b')[-1].split("^")[1][-1] == 'H':
                sex = "F"
            else:
                sex = "F"
            #print("AFTER IPOLOGISMOS: ",sex)


        # if str(name.original_string).split('b')[-1].split("^")[1][0] == 'AGNOSTON' and str(name.original_string).split('b')[-1].split("^")[1][-1] == 'S'
        result_dict = {
            'slices': windowed_slices,
            'name': name,
            'sex': sex,
            'age': age,
            'Glasgow Coma Scale' : GCS,
            'Class' :CLASS
        }
        return result_dict
    
    def _resize_volume(self,img):
        print("The actual shape of the scan is:",img.shape)
        """Resize across z-axis"""
        # Set the desired depth
        desired_depth, desired_width, desired_height= self.resamling_dimensions
        current_depth = img.shape[0]
        current_width = img.shape[1]
        current_height = img.shape[2]
        # Compute depth factor
        depth = current_depth / desired_depth
        width = current_width / desired_width
        height = current_height / desired_height
        depth_factor = 1 / depth
        width_factor = 1 / width
        height_factor = 1 / height
        # # Rotate
        # img = ndimage.rotate(img, 90, reshape=False)
        # Resize across z-axis
        img = ndimage.zoom(img, (depth_factor,width_factor, height_factor, ), order=5)
        # img = ndimage.zoom(img, (depth_factor ,width_factor, height_factor), order=1)
        return img
    
    def process_scan(self,path_of_dicom_slices_from_one_patient,folder_of_patient_name):

        def min_max_normalize(array):
            min_val = array.min()
            max_val = array.max()
            # Apply min-max normalization
            normalized_array = (array - min_val) / (max_val - min_val)
            return normalized_array
        
        def set_zero_to_image(volume):return volume
            # counts, bins = np.histogram(volume, bins=10)
            # bin_max_index =np.argmax(counts)
            # volume[volume<=bins[bin_max_index+1]]=0
            # return volume
        
        """Read and resize volume"""
        # Read scan
        
        volume = self._read_dicom_images(path_of_dicom_slices_from_one_patient,folder_of_patient_name)['slices']
        name = self._read_dicom_images(path_of_dicom_slices_from_one_patient,folder_of_patient_name)['name']
        sex = self._read_dicom_images(path_of_dicom_slices_from_one_patient,folder_of_patient_name)['sex']
        age = self._read_dicom_images(path_of_dicom_slices_from_one_patient,folder_of_patient_name)['age']
        if age!='000Y':
            age = re.sub(r'^0+', '', re.sub(r'\D', '', age))
        elif age=='000Y':
            age="NA"
        print("the type of scan is: {}".format(SERIES))
        print("WL:",window_center,"WW:", window_width)
        # volume = normalize(volume)
        # Resize width, height and depth
        volume =  self._resize_volume(volume)

        central_slice = volume.shape[0]//2
        from find_midline import midline
        from segment_brain import segment
        THRESHOLD = 20
        thresholded = volume*255>THRESHOLD
        mask = np.vectorize(segment, signature='(n,m)->(n,m)')(thresholded)
###########################################################################################################
        print("the initial angle is: ",midline(mask[central_slice],name))
        
        if midline(mask[central_slice],name,graph=False)<-14:
            print("procceding rotation... <-14 degrees")
            from align import align_image
            # volume_aligned = np.vectorize(align_image, signature='(n,m)->(n,m)')(volume)
            volume_aligned=[]
            angle = midline(mask[central_slice],name,graph=False)
            print("angle in perpenticular plane: {}".format(angle))
            for slice in volume*255:
                volume_aligned.append(align_image(slice,angle))
            volume_aligned = np.array(volume_aligned)
            # volume = align_image(volume,midline(mask[central_slice],name,graph=False))
            print("patient's name:{},patient's sex:{},patients'age:{},Glasgow Coma Scale:{},Class:{}".format(name,sex,age,GCS,CLASS))
            print(set_zero_to_image( min_max_normalize(volume_aligned)).min(),set_zero_to_image( min_max_normalize(volume_aligned)).max())
            return {'volume': set_zero_to_image( min_max_normalize(volume_aligned)),
            'name':name,
            'sex':sex,
            'age':age,
            'Glasgow Coma Scale' : GCS,
            'Class':CLASS
            }
        

        if midline(mask[central_slice],name,graph=False)>90:
            print("procceding rotation... >90 degrees")
            from align import align_image
            # volume_aligned = np.vectorize(align_image, signature='(n,m)->(n,m)')(volume)
            volume_aligned=[]
            angle = midline(mask[central_slice],name,graph=False)
            angle = angle-180
            print("angle in perpenticular plane: {}".format(angle))
            for slice in volume:
                volume_aligned.append(align_image(slice,angle))
            volume_aligned = np.array(volume_aligned)
            # volume = align_image(volume,angle)
            print("patient's name:{},patient's sex:{},patients'age:{},Glasgow Coma Scale:{},Class:{}".format(name,sex,age,GCS,CLASS))
            print(set_zero_to_image( min_max_normalize(volume_aligned)).min(),set_zero_to_image( min_max_normalize(volume_aligned)).max())
            return {'volume': set_zero_to_image( min_max_normalize(volume_aligned)),
            'name':name,
            'sex':sex,
            'age':age,
            'Glasgow Coma Scale' : GCS,
            'Class':CLASS
            }

        if midline(mask[central_slice],name,graph=False)>14 and midline(mask[central_slice],name,graph=False)<90:
            print("procceding rotation... >14  and <90 degrees")
            from align import align_image
            # volume_aligned = np.vectorize(align_image, signature='(n,m)->(n,m)')(volume)
            volume_aligned=[]
            angle = midline(mask[central_slice],name,graph=False)
            print("angle in perpenticular plane: {}".format(angle))
            for slice in volume:
                volume_aligned.append(align_image(slice,angle))
            volume_aligned = np.array(volume_aligned)
            # volume = align_image(volume,midline(mask[central_slice],name,graph=False))
            print("patient's name:{},patient's sex:{},patients'age:{},Glasgow Coma Scale:{},Class:{}".format(name,sex,age,GCS,CLASS))
            print(set_zero_to_image( min_max_normalize(volume_aligned)).min(),set_zero_to_image( min_max_normalize(volume_aligned)).max())
            return {'volume': set_zero_to_image( min_max_normalize(volume_aligned)),
            'name':name,
            'sex':sex,
            'age':age,
            'Glasgow Coma Scale' : GCS,
            'Class':CLASS
            }

                
        
        # volume =  resample(volume)  ########################## DATALORE
        print("patient's name:{},patient's sex:{},patients'age:{},Glasgow Coma Scale:{},Class:{}".format(name,sex,age,GCS,CLASS))
        print("-"*100)
        print(set_zero_to_image( min_max_normalize(volume)).min(),set_zero_to_image( min_max_normalize(volume)).max())
        return {'volume': set_zero_to_image( min_max_normalize(volume)),
                'name':name,
                'sex':sex,
                'age':age,
                'Glasgow Coma Scale' : GCS,
                'Class':CLASS
                }
    
