# -*- coding: utf-8 -*-
"""
@author: Negmo
"""
from skimage import io, img_as_ubyte
import os
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from sklearn import svm,tree

#load an image from a given directory
def load(imdir,f):
    io.use_plugin('pil')
    image=io.imread(os.path.join(imdir, f),True)
    return img_as_ubyte(image)
    
train_size=10
   
#collect all the stones in an array
stones=[]
for typ in 'abcd':
    for i in range(0,train_size):
        stones.append(load(r"\dataset\%s" % typ,'%d.jpg' % i))#change this directory
        
#extract texture features using glcm
xs=[]#array of dissimilarities
ys=[]#array of correlation
zs=[]#array of ASM
for i in range(len(stones)):
    glcm = greycomatrix(stones[i], [5], [0], 256, symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])
    zs.append(greycoprops(glcm,'ASM')[0,0])
    
#classification    
clar=[]
cltype=[]
for i in range(len(xs)):
        clar.append([xs[i],ys[i],zs[i]])
        cltype.append("abcd"[i//train_size])

svm_clf=svm.SVC()
svm_clf.fit(clar,cltype)

tree_clf=tree.DecisionTreeClassifier()
tree_clf.fit(clar,cltype)

#testing predictions
svm_incorrect=[]
svm_correct=[]

tree_incorrect=[]
tree_correct=[]
for typ in 'abcd':
    for i in range(train_size,37):
        p_img=load(r"\dataset\%s" % typ,'%d.jpg' % i)#change this directory 
        p_glcm=greycomatrix(p_img, [5], [0], 256, symmetric=True, normed=True)
        p=[greycoprops(p_glcm, 'dissimilarity')[0, 0],greycoprops(p_glcm, 'correlation')[0, 0],greycoprops(p_glcm,'ASM')[0,0]]
        svm_prediction=svm_clf.predict([p])
        tree_prediction=tree_clf.predict([p])
        if typ!=svm_prediction:
            svm_incorrect.append((typ,svm_prediction[0],i))
        else:
            svm_correct.append((typ,i))
            
        if typ!=tree_prediction:
            tree_incorrect.append((typ,tree_prediction[0],i))
        else:
            tree_correct.append((typ,i))
            
svm_precision=(((37-train_size)*4-len(svm_incorrect))/((37-train_size)*4))*100
tree_precision=(((37-train_size)*4-len(tree_incorrect))/((37-train_size)*4))*100
# for each type, plot (dissimilarity, correlation,asm)
fig = plt.figure(figsize=(20, 20))            
ax = fig.add_subplot(3, 1, 2)
for i in range(4):
    ax.plot(xs[train_size*i:train_size*(i+1)],[10**i]*train_size, 'go')
for i in range(4):
    ax.plot(ys[train_size*i:train_size*(i+1)], [10**i]*train_size,'bo')
for i in range(4):
    ax.plot(zs[train_size*i:train_size*(i+1)],[10**i]*train_size, 'ro')


ax.set_xlabel('GLCM features')
ax.set_ylabel('stone types')
ax.legend()
fig.suptitle('Grey level co-occurrence matrix features \n precision :\n disicion tree:%f \n svm:%f' %(tree_precision,svm_precision), fontsize=14)
plt.show()
