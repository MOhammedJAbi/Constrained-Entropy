
import os
import glob
from os.path import isfile, join
from PIL import Image
import numpy as np
import nibabel as nib
import scipy.io 
import pdb
from PIL import Image
from skimage import io

from skimage.measure import regionprops

from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from scipy import ndimage
from skimage import morphology, measure

from skimage import color
from skimage import io


if __name__ == '__main__':

    cases=["case_4","case_10","case_14","case_29","case_58","case_73","case_74","case_78","case_81"]
    frames=[8,4,4,4,2,2,2,2,22]
    
    #cases=["case_1","case_16","case_22","case_23","case_34","case_57","case_82","case_83","case_84","case_88","case_92"]
    #frames=[8,4,2,4,2,2,16,16,16,16,16]
    sortedlist=[]
    for frame,case in zip(frames,cases):
        images=[]
        image_numpy = np.zeros((256,256,frame))
        for i in range(frame):
            #mat=scipy.io.loadmat("../Results/Images_MATLAB/FusionNetModified_32_LateFusion_14/"+case+"_"+str(i)+ '.mat')
            #mat=scipy.io.loadmat("../Results/Images_MATLAB/LateFusion_4Mod/"+case+"_"+str(i)+ '.mat')
            #mat=scipy.io.loadmat("../Results/Images_MATLAB/FusionNetModified_LateFusion_2/"+case+"_"+str(i)+ '.mat')
            #mat=scipy.io.loadmat("../Results/Images_MATLAB/FusionNetModified_HD_asym_2Shuffle_Upsample/"+case+"_"+str(i)+ '.mat')
            #mat=scipy.io.loadmat("../Results/Images_MATLAB/FusionNetModified_HD_asym_4Mod/"+case+"_"+str(i)+ '.mat')
            #mat=scipy.io.loadmat("../Results/Images_MATLAB/FusionNetModified_HD_asym_2/"+case+"_"+str(i)+ '.mat')
            #mat=scipy.io.loadmat("../Results/Images_MATLAB/ERFNet/"+case+"_"+str(i)+ '.mat')
            #mat=scipy.io.loadmat("../Results/Images_MATLAB/FusionNetModified_HD_asym_2Shuffle/"+case+"_"+str(i)+ '.mat')
            #mat=scipy.io.loadmat("../Results/Images_MATLAB/UNet_NewData/"+case+"_"+str(i)+ '.mat')
            
            #slice_numpy = color.rgb2gray(io.imread("../Results/Images_PNG/FusionNetModified_LateFusion_val_6/"+case+"_"+str(i)+ '.png'))
            
            #slice_numpy = Image.open().convert('LA')
            #img_array = np.asarray(slice_numpy)
            
            #img2D=scipy.io.loadmat("../Results/Images_PNG/UNet_Original"+case+"_"+str(i)+ '.png')
            
            #pdb.set_trace()
           
            #slice_numpy = np.array(mat['pred'])
            #
            slice_numpy = scipy.misc.imread("../Data/ISLES_png/val/MTT/"+case+"_"+str(i)+ '.png')
            
            image_numpy[:,:,i] = slice_numpy

        #pdb.set_trace()


        struct = generate_binary_structure(3, 3)
        labeled_array, num_regions = label(image_numpy, structure=struct)
        sizes = ndimage.sum(image_numpy, labeled_array, range(1, num_regions + 1))
        for i in range(len(sizes)):
            print('{} '.format(sizes[i]))
        map = np.where(sizes >10)[0] + 1
   
        max_index = np.zeros(num_regions + 1, np.uint8)
        max_index[map] = 1
        cleanedImage = np.zeros(image_numpy.shape)
        cleanedImage = max_index[labeled_array]
            
        #modelName = 'FusionNetModified_LateFusion_2'
        modelName = 'MTTJose'
        #modelName = 'FusionNetModified_HD_asym_2Shuffle_Upsample'
        #modelName = 'FusionNetModified_HD_asym_2'
        #modelName = 'FusionNetModified_HD_asym_4Mod'
        #modelName = 'FusionNetModified_HD_asym_2Shuffle'
        #modelName = 'FusionNetModified_HD'
        directory = '../Results/' + modelName 
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        niftiName= directory+'/Test' + case+'.nii'
        
        xform = np.eye(4) 
        #imgNifti = nib.nifti1.Nifti1Image(cleanedImage, xform)
        imgNifti = nib.nifti1.Nifti1Image(image_numpy, xform)
        nib.save(imgNifti, niftiName)


