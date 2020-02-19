# Classification-of-stones
This project aims at identifying different types of stones using texture features

## 1.DATASET :
The used data set can be downloaded from here : http://bayanbox.ir/id/3168748744619895304?download
It includes 37 images for 4 types of stones (108 image in total,6.3 MB)
(change the directory of dataset into your dataset directory)
  
## 2.PROJECT FILES :
dissimilarity.py : classifying stones only based on dissimilarity calculated from GLCM
asm.py : classifying stones only based on asm calculated from GLCM
correlation.py : classifying stones only based on correlation calculated from GLCM
3-feature.py : classifying stones based on asm,dissimilarity,correlation calculated from GLCM
  
## 3.INSTALLING PREREQUISTIES : 
Python 3.5 
You need to install sklearn,skimage,matplotlib modules of python

