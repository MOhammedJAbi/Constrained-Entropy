""" 
Copyright (c) 2016, Jose Dolz .All rights reserved.

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

Jose Dolz. Dec, 2016.
email: jose.dolz.upv@gmail.com
LIVIA Department, ETS, Montreal.
"""

import numpy as np
import pdb
import nibabel as nib
import scipy.io as sio


# ----- Loader for nifti files ------ #
def load_nii (imageFileName, printFileNames) :
    if printFileNames == True:
        print (" ... Loading file: {}".format(imageFileName))

    img_proxy = nib.load(imageFileName)
    imageData = img_proxy.get_data()
    
    return (imageData,img_proxy)
    
def release_nii_proxy(img_proxy) :
    img_proxy.uncache()


# ----- Loader for matlab format ------- #
# Very important: All the volumes should have been saved as 'vol'.
# Otherwise, change its name here
def load_matlab (imageFileName, printFileNames) :
    if printFileNames == True:
        print (" ... Loading file: {}".format(imageFileName))
    
    mat_contents = sio.loadmat(imageFileName)
    imageData = mat_contents['predVol']
    
    return (imageData)
    
""" It loads the images (CT/MRI + Ground Truth + ROI) for the patient image Idx"""
