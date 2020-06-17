import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from skimage.transform import resize
from tifffile import imsave
from PIL import Image
import bioformats
import re
from pathlib import Path
from numpy import asarray



def correct_file(location, patient, phenotypes, file):
    """ 
    correct_file: returns true if tiff file has the correct location,
    patient_id and is in the desired phenotype list

    @input: 
    - str location: location of patch
    - str patient: patient_id
    - list[str] phenotypes: list of phenotypes wanted
    - str file: file name

    @output: Bool, see above
    """
    
    if (file[-3:] == 'tif') and (patient in file) and (location in file):
        for phenotype in phenotypes:
            if phenotype in file:
                return True
        return False
    else:
        return False
    
def create_grayscale(path_hr, path_lr, file, new_size = (1004, 1344, 3), to_resize = False):
    """
    create_grayscale: Creates a one channel resized image of a phenotype output of inform
    @input: 
    - str path_hr: path to the lcoation of original HR phenotype patches
    - str path_lr: path to the lcoation of resized HR phenotype patches 
        (only necessary if we want to save the 1 channel grayscale images, then uncomment below)
    - str file: file name of phenotype to be resized
    - tuple new_size: desired size
    - bool to_resize: true if image needs to be resized to `new_size`

    @output: np.array see above
    """
    #Resize HR patch to LR patch size of (500, 669):

    #load image:
    hr_patch =  plt.imread(path_hr + file)
    assert (hr_patch.shape == (1004, 1344, 3))
    
    if to_resize: 
        #resize image:
        hr_patch_resized = resize(hr_patch, new_size, anti_aliasing=True)
        print(hr_patch_resized)
        assert (list(hr_patch_resized.shape) == list(new_size))
    else: hr_patch_resized = hr_patch

    grayscale_image = Image.fromarray(hr_patch_resized)
    grayscale_image = grayscale_image.convert('L')

    return grayscale_image

def create_train_data(patients,
                      phenotypes,
                      path_hr,
                      path_lr,
                      resize_size=(1004, 1344, 3),
                      to_resize=True):
    """
    create_train_data: creates the 6 channel images (either resized or not) depending on the inout
    @input: 
    - [str] patients: patient_ids
    - list[str] phenotypes: list of phenotypes wanted
    - str path_hr: path to the location of original HR phenotype patches
    - str path_lr: path to the location of our new resized HR patch
    - tuple resize_size: desired size (if not given then we will not resize)
    - bool to_resize: true if the image has to be resized
    
    @output: saves new images in path_lr
    """
    print('Saving to: ', path_lr)

    # First delete content of directory if not empty:
    p = Path(path_lr)
    for f in p.glob('*.tif'):
        try:
            f.unlink()
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

    for patient in patients:
        print('Patient: ' + patient)
        locations = get_locations(path_hr, patient)
        print('Locations for patient ' + patient + ' :' + str(locations))
        for location in locations:
            arr = multichannel_phenotype(patient=patient,
                                         location=location,
                                         phenotypes=phenotypes,
                                         path_hr=path_hr,
                                         path_lr=path_lr,
                                         to_resize=to_resize,
                                         new_size=resize_size)
            #save new image:
            assert (arr.size != 0)
            
            # Divide image into 4 parts:
            s = list(resize_size)
            height, width = arr[0,:,:].shape

            s[0] = len(phenotypes)
            s[1] = int(height/2)
            s[2] = int(width/2)            
            im1 = np.zeros(tuple(s), dtype='uint8')
            im2 = np.zeros(tuple(s), dtype='uint8')
            im3 = np.zeros(tuple(s), dtype='uint8')
            im4 = np.zeros(tuple(s), dtype='uint8')
            
            for i in range(len(phenotypes)):
                im = arr[i, :, :]
                im1[i,:,:] = im[0:int((height / 2)), 0:int((width / 2))]
                im2[i,:,:] = im[0:int((height / 2)), width-int((width / 2)):width]
                im3[i,:,:] = im[height-int((height / 2)):height, 0:int((width / 2))]
                im4[i,:,:] = im[height-int((height / 2)):height, width-int((width / 2)):width]
    
                
            imsave(path_lr + patient + '_' + location + '_part_1_1'+'_.tif', im1[0:3,:,:].transpose(1,2,0),dtype='uint8')
            imsave(path_lr + patient + '_' + location + '_part_2_1'+'_.tif', im2[0:3,:,:].transpose(1,2,0),dtype='uint8')
            imsave(path_lr + patient + '_' + location + '_part_3_1'+'_.tif', im3[0:3,:,:].transpose(1,2,0),dtype='uint8')
            imsave(path_lr + patient + '_' + location + '_part_4_1'+'_.tif', im4[0:3,:,:].transpose(1,2,0),dtype='uint8')
            imsave(path_lr + patient + '_' + location + '_part_1_2'+'_.tif', im1[3:,:,:].transpose(1,2,0),dtype='uint8')
            imsave(path_lr + patient + '_' + location + '_part_2_2'+'_.tif', im2[3:,:,:].transpose(1,2,0),dtype='uint8')
            imsave(path_lr + patient + '_' + location + '_part_3_2'+'_.tif', im3[3:,:,:].transpose(1,2,0),dtype='uint8')
            imsave(path_lr + patient + '_' + location + '_part_4_2'+'_.tif', im4[3:,:,:].transpose(1,2,0),dtype='uint8')

        print('Finished patient:' + patient)
    print('Done with all patients')

def find_phen_files(patient, phenotype,hr_path):
    """
    find_phen_files: finds the files for a certain phenotype and patient
    @input: 
    - str patient: patient id
    - str phenotype: phenotype
    - str hr_path: path to phenotype files
    """
    
    with os.scandir(mypath) as entries:
        files = [entry.name for entry in entries if entry.is_file()]

    patient_phen_files = [file for file in files if file[:4] == patient and bool(re.search(phenotype, file))]

    print('Files available for that patient and phenotype:')
    print(patient_phen_files)
    return patient_phen_files


def get_locations(path_hr, patient):
    """
    get_locations: gives the different HR patch locations for a patient
    @input: 
    - str path_hr: path to the HR patches
    - str patient: patient id
    @output: [str] see above
    """
    with os.scandir(path_hr) as entries:
        files = [entry.name for entry in entries if entry.is_file()]
    
    patient_phen_files = [file for file in files if file[:4] == patient]
    locations = []
    for file in patient_phen_files:
        pat = r'.*?\[(.*)].*'
        match = re.search(pat, file)
        match = match.group(1)
        locations.append('[' + str(match) + ']')
    return np.unique(np.array(locations))

def multichannel_phenotype(patient,
                           location,
                           phenotypes,
                           path_hr,
                           path_lr,
                           to_resize=False,
                           new_size=(1004, 1344, 3)):
    """
    multichannel_phenotype: creates a 6 channel image for a location of an HR patch
    each channel is a phenotype output of inform

    @input: 
    - str location: location of patch
    - str patient: patient_id
    - list[str] phenotypes: list of phenotypes wanted
    - tuple new_size: desired size
    - str path_hr: path to the location of original HR phenotype patches
    - str path_lr: path to the location of our new resized HR patch

    @output: (500, 669, 6) array of a HR phenotype patch
    """

    #Find all HR phenotype patches:
    with os.scandir(path_hr) as entries:
        files = [entry.name for entry in entries if entry.is_file()]

    #Get all files for a patient and certain location:
    patient_loc_HR = [
        file for file in files
        if correct_file(location, patient, phenotypes, file)
    ]

    if len(patient_loc_HR) != len(phenotypes):
        print(patient_loc_HR)
        assert False

    #Resize the phenotype files and create one image with 6 channels(one for each phenotype):

    #Transform the size to 6 channels:
    s = list(new_size)
    s[0] = len(patient_loc_HR)
    s[1] = new_size[0]
    s[2] = new_size[1]
    arr = np.zeros(tuple(s), dtype='uint8')

    #Create a one channel image for each phenotype and concatenate them:
    for i in range(len(patient_loc_HR)):
        im = create_grayscale(path_hr, path_lr, patient_loc_HR[i],new_size, to_resize)
        arr[i, :, :] = asarray(im)
    assert (arr.size != 0)

    return arr

def show_6_chann_phen(multi_chann_array, phenotypes):
    """
    show_6_chann_phen: shows the 6 channels of our created images
    @input: 
    - np.arary multi_chann_array: HR resized patch
    - list[str] phenotypes: list of phenotypes wanted
    """
    
    #Print it for one location:
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))

    for i in range(6):
        if i < 3:
            ax[0, i].imshow(multi_chann_array[i,:, :])
            ax[0, i].set_title('Phenotype: ' + phenotypes[i])
        else:
            ax[1, i - 3].imshow(multi_chann_array[i,:, :])
            ax[1, i - 3].set_title('Phenotype: ' + phenotypes[i])

    fig.show()
    