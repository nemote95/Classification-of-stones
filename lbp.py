# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 19:17:46 2016

@author: Negmo
"""

"""
===============================================
Local Binary Pattern for texture classification
===============================================

"""
import numpy as np
from skimage.feature import local_binary_pattern
from skimage import io
from skimage import img_as_ubyte
from skimage.filters import gabor_filter
from sklearn import svm,tree
import os 

# settings for LBP

METHOD = 'uniform'

radius = 1
n_points = 10 * radius

#load an image from a given directory
def load(imdir,f):
    io.use_plugin('pil')
    image=io.imread(os.path.join(imdir, f),True)
    real,img=gabor_filter(image,frequency=3.5)
    return img_as_ubyte(image)
    
train_size=10
   
#collect all the stones in an array
clar=[]
cltype=[]
for typ in 'abcd':
    for i in range(0,train_size):
        img=load(r"C:\Users\Negmo\.spyder2-py3\dataset\%s" % typ,'%d.jpg' % i)
        lbp=local_binary_pattern(img, n_points, radius, METHOD)
        n_bins = lbp.max() + 1
        his,_=np.histogram(lbp, normed=True, bins=n_bins,range=(0, n_bins))
        clar.append(his)
        cltype.append(typ)

# classify rotated textures
svm_clf=svm.SVC()
svm_clf.fit(clar,cltype)
tree_clf=tree.DecisionTreeClassifier()
tree_clf.fit(clar,cltype)
correct_svm=0
correct_tree=0
for typ in 'abcd':
    for i in range(train_size,37):
        img=load(r"C:\Users\Negmo\.spyder2-py3\dataset\%s" % typ,'%d.jpg' % i)
        lbp=local_binary_pattern(img, n_points, radius, METHOD)
        n_bins = lbp.max() + 1
        his,_=np.histogram(lbp, normed=True, bins=n_bins,range=(0, n_bins))
        rec_svm=svm_clf.predict([his])
        rec_tree=tree_clf.predict([his])
        if typ==rec_svm[0]:
            correct_svm+=1
        if typ==rec_tree[0]:
            correct_tree+=1
print (correct_svm,correct_tree)
            
