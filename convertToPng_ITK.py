"""
Copyright (c) 2017, Jose Dolz .All rights reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
    OTHER DEALINGS IN THE SOFTWARE.
Jose Dolz. Dec, 2017.
email: jose.dolz.upv@gmail.com
LIVIA Department, ETS, Montreal.
"""


import sys
import pdb
from os.path import isfile, join
import os
import numpy as np
import nibabel as nib
import scipy.io as sio
import numpngw

from skimage.measure import regionprops
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from scipy import ndimage

from loadData import load_nii

import SimpleITK as sitk
from numpy.core.umath_tests import inner1d

from random import random, randint

from PIL import Image, ImageOps
# python evaluate.py ./ResultsNifti/ENet/

""" To print function usage """

#python convertToPng_ITK.py ../Data/TRAINING/ ../Data/ISLES_png_Jose/

def printUsage(error_type):
    if error_type == 1:
        print(" ** ERROR!!: Few parameters used.")
    else:
        print(" ** ERROR!!: ...")  # TODO

    print(" ******** USAGE ******** ")
    print(" --- argv 1: Folder containing CNN results")



def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
        imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]

    imageNames.sort()

    return imageNames


def convertToPng(argv):
    # Number of input arguments
    #    1: Folder containing label images

    if len(argv) < 1:
        printUsage(1)
        sys.exit()

    imagesFolder = argv[1]
    print(imagesFolder)
    
    imagesFolderDst = argv[2]
    print(imagesFolderDst)
    '''if not os.path.exists(imagesFolderDst+'/GT'):
            os.makedirs(imagesFolderDst+'/GT')
    
    if not os.path.exists(imagesFolderDst+'/CBV'):
            os.makedirs(imagesFolderDst+'/CBV')

    if not os.path.exists(imagesFolderDst+'/CBF'):
            os.makedirs(imagesFolderDst+'/CBF')
                 
    if not os.path.exists(imagesFolderDst+'/CT'):
            os.makedirs(imagesFolderDst+'/CT')
            
    if not os.path.exists(imagesFolderDst+'/DWI'):
            os.makedirs(imagesFolderDst+'/DWI')
            
    if not os.path.exists(imagesFolderDst+'/MTT'):
            os.makedirs(imagesFolderDst+'/MTT')

    if not os.path.exists(imagesFolderDst+'/Tmax'):
            os.makedirs(imagesFolderDst+'/Tmax')'''

    if not os.path.exists(imagesFolderDst+'/OT'):
            os.makedirs(imagesFolderDst+'/OT')
    
    if not os.path.exists(imagesFolderDst+'/CT_4DPWI'):
            os.makedirs(imagesFolderDst+'/CT_4DPWI')

    if not os.path.exists(imagesFolderDst+'/CT_CBF'):
            os.makedirs(imagesFolderDst+'/CT_CBF')
                 
    if not os.path.exists(imagesFolderDst+'/CT_CBV'):
            os.makedirs(imagesFolderDst+'/CT_CBV')
            
    if not os.path.exists(imagesFolderDst+'/CT_MTT'):
            os.makedirs(imagesFolderDst+'/CT_MTT')
            
    if not os.path.exists(imagesFolderDst+'/CT_Tmax'):
            os.makedirs(imagesFolderDst+'/CT_Tmax')

    if not os.path.exists(imagesFolderDst+'/CT'):
            os.makedirs(imagesFolderDst+'/CT')
            
                    
    subjectNames = os.listdir(imagesFolder)
    subjectNames.sort()

    
    printFileNames = True
    for s_i in range(len(subjectNames)):
        #imageNames = getImageImageList(imagesFolder+subjectNames[s_i]+'/')
        imageNames = os.listdir(imagesFolder+'/'+subjectNames[s_i]+'/')
        namesSave=['MTT','CBF','Tmax','GT','CBV','DWI','CT']
        for f_i in range(len(imageNames)):

            print(imageNames[f_i])
            #pdb.set_trace()
            #imageData_ITK = sitk.ReadImage(imagesFolder+subjectNames[s_i]+'/'+imageNames[f_i]+'/'+imageNames[f_i]+'.nii')
            #imageData = sitk.GetArrayFromImage(imageData_ITK)
            
            [imageData, img_proxy] = load_nii(imagesFolder+'/'+subjectNames[s_i]+'/'+imageNames[f_i]+'/'+imageNames[f_i]+'.nii', printFileNames)
            
            imageData_Norm = 65535*((imageData- imageData.min())/imageData.ptp())
            imageData_Norm = imageData_Norm.astype(np.uint16)

            names = imageNames[f_i].split('.O.')
            names = names[1].split('.')
            
            if (len(imageData.shape) < 4):
                for i_i in range(imageData_Norm.shape[2]):
                #for i_i in range(imageData_Norm.shape[0]):
                    img2D = imageData_Norm[:,:,i_i]
                    #img2D = imageData_Norm[i_i,:,:]
                
                    numpngw.write_png(imagesFolderDst+'/'+names[0]+'/'+subjectNames[s_i]+'_'+str(i_i)+'.png', img2D)
            else:
                #pdb.set_trace()
                for i_i in range(imageData_Norm.shape[2]):
                #for i_i in range(imageData_Norm.shape[0]):
                    img2D = imageData_Norm[:,:,i_i,0]
                    #img2D = imageData_Norm[i_i,:,:]
                
                    numpngw.write_png(imagesFolderDst+'/'+names[0]+'/'+subjectNames[s_i]+'_'+str(i_i)+'.png', img2D)
                    
        #pdb.set_trace()
    print(" ******************************************  PROCESSING LABELS DONE  ******************************************")


if __name__ == '__main__':
    convertToPng(sys.argv)
