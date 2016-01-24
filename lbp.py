"""
@author: Negmo
"""
import numpy as np
from skimage.feature import local_binary_pattern
from skimage import io
from skimage.filters import gabor_filter
from sklearn import svm,tree
import matplotlib.pyplot as plt
import os 

# settings for LBP

METHOD = 'uniform'

radius = 1
n_points = 10 * radius

train_size=10

#load an image from a given directory
def load(imdir,f):
    io.use_plugin('pil')
    image=io.imread(os.path.join(imdir, f),True)
    real,img=gabor_filter(image,frequency=7.5)
    return image
    
   
#collect all the stones lbp histogram in an array
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

# classify other stones

svm_clf=svm.SVC()
svm_clf.fit(clar,cltype)
tree_clf=tree.DecisionTreeClassifier()
tree_clf.fit(clar,cltype)
correct_svm=0
correct_tree=0
print('expectation','prediction')
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
        else :print(typ,'\t',rec_svm[0])
        if typ==rec_tree[0]:
            correct_tree+=1
        #else print(typ,rec_tree[0])
print ("svm_precision:",correct_svm/(4*(37-train_size)),"tree_precision:",correct_tree/(4*(37-train_size)))

# plot histograms of LBP of textures
def hist(ax, lbp):
    n_bins = lbp.max() + 1
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),facecolor='0.5')
   
fig, ((ax1, ax2, ax3,ax7), (ax4, ax5, ax6,ax8)) = plt.subplots(nrows=2, ncols=4,
                                                       figsize=(12, 9))
plt.gray()                

img=load(r"C:\Users\Negmo\.spyder2-py3\dataset\a",'0.jpg')
ax1.imshow(img)
ax1.axis('off')
lbp=local_binary_pattern(img, n_points, radius, METHOD)
hist(ax4, lbp)
ax4.set_ylabel('Percentage')


img=load(r"C:\Users\Negmo\.spyder2-py3\dataset\b",'0.jpg')
ax2.imshow(img)
ax2.axis('off')
lbp=local_binary_pattern(img, n_points, radius, METHOD)
hist(ax5, lbp)

img=load(r"C:\Users\Negmo\.spyder2-py3\dataset\c",'0.jpg')
ax3.imshow(img)
ax3.axis('off')
lbp=local_binary_pattern(img, n_points, radius, METHOD)
hist(ax6, lbp)


img=load(r"C:\Users\Negmo\.spyder2-py3\dataset\d",'26.jpg')
ax7.imshow(img)
ax7.axis('off')
lbp=local_binary_pattern(img, n_points, radius, METHOD)
hist(ax8, lbp)


ax5.set_xlabel('Uniform LBP values')
