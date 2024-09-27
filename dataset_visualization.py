import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


patients1 = np.load("/raid/theodoropoulos/PhD/Data/128x128x120/Train/patients_Train.npy",allow_pickle=True)
patients2 = np.load("/raid/theodoropoulos/PhD/Data/128x128x120/Test/patients_Test.npy",allow_pickle=True)
patients = np.concatenate((patients1,patients2),axis=0)



# Initialize lists to store the data
sex = []
age = []
glasgow_coma_scale = []
patient_class = []

# Extract the data from each dictionary in the array
for patient_data in patients:
    sex.append(patient_data['sex'])
    
    # Handle 'NA' for age and convert to integer or use np.nan
    if patient_data['age'] == "NA":
        age.append(np.nan)
    else:
        age.append(int(patient_data['age']))
    
    # Handle 'NA' for Glasgow Coma Scale and convert to integer or use np.nan
    if patient_data['Glasgow Coma Scale'] == "NA":
        glasgow_coma_scale.append(np.nan)
    else:
        glasgow_coma_scale.append(int(patient_data['Glasgow Coma Scale']))
    
    # Class should always be an integer, so no need for special handling
    patient_class.append(int(patient_data['Class']))

# Create a DataFrame for easier plotting
df = pd.DataFrame({
    'Sex': sex,
    'Age': age,
    'Glasgow Coma Scale': glasgow_coma_scale,
    'Class': patient_class
})

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Sex distribution
sns.countplot(x='Sex', data=df, palette='Set2', ax=axs[0, 0])
axs[0, 0].set_title('Distribution of Sex')

# Age distribution
sns.histplot(df['Age'].dropna(), bins=10, kde=True, color='skyblue', ax=axs[0, 1])
axs[0, 1].set_title('Distribution of Age')

# Glasgow Coma Scale distribution
sns.histplot(df['Glasgow Coma Scale'].dropna(), bins=15, kde=False, color='lightgreen', ax=axs[1, 0])
axs[1, 0].set_title('Distribution of Glasgow Coma Scale')

# Class distribution
sns.countplot(x='Class', data=df, palette='Set3', ax=axs[1, 1])
axs[1, 1].set_title('Distribution of Classes')

plt.tight_layout()
plt.show()
plt.savefig('/raid/theodoropoulos/PhD/Results/Dataset_visualization.png')