from sg import sg
from pylab import figure,pcolor,scatter,contour,colorbar,show,imshow
from numpy import meshgrid,reshape,linspace,ones,min,max,concatenate,transpose
from numpy import ravel,array
from numpy.random import randn
import numpy

num=200;
sg('send_command', 'loglevel ALL')
features=concatenate((randn(2,num)-1,randn(2,num)+1),1)
labels=concatenate((-ones([1,num]), ones([1,num])),1)[0]
sg('set_features', "TRAIN", features)
sg('set_labels', "TRAIN", labels)
sg('send_command', 'set_kernel GAUSSIAN REAL 10 5')
sg('send_command', 'init_kernel TRAIN')
sg('send_command', 'new_svm GPBTSVM')
sg('send_command', 'c 100')
sg('send_command', 'train_classifier')
[bias, alphas]=sg('get_svm');
print bias
print alphas
figure()

x1=linspace(1.2*min(features),1.2*max(features), 50)
x2=linspace(1.2*min(features),1.2*max(features), 50)
x,y=meshgrid(x1,x2);
testfeatures=array((ravel(x), ravel(y)))

sg('set_features', "TEST", testfeatures)
sg('send_command', 'init_kernel TEST')
z=sg('svm_classify')

z.resize((50,50))
#pcolor(x, y, transpose(z), shading='flat') #for non-smooth visualization
i=imshow(transpose(z),  origin='lower', extent=(1.2*min(features),1.2*max(features),1.2*min(features),1.2*max(features))) #for smooth visualization
scatter(features[0,:],features[1,:], s=20, marker='o', c=labels, hold=True)
contour(x, y, transpose(z), linewidths=1, colors='black', hold=True)
colorbar(i)
show()
