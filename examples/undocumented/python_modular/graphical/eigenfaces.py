#!/usr/bin/env python

#Copyright (c) The Shogun Machine Learning Toolbox
#Written (W) 2014 Alejandro Hernandez Cordero
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#The views and conclusions contained in the software and documentation are those
#of the authors and should not be interpreted as representing official policies,
#either expressed or implied, of the Shogun Development Team.

#This example is about learning and using eigenfaces in Shogun.
#We demonstrate how to use them for a set of faces.

#OpenCV must be installed
try:
    import cv2
except ImportError:
    print 'Error: OpenCV must be installed'
    exit()

import numpy as np
from numpy    import random

from modshogun import RealFeatures
from modshogun import PCA
from modshogun import EuclideanDistance
import math
import os
import pylab as pl

IMAGE_WIDHT = 80
IMAGE_HEIGHT = 80
N_SUBSET = 10

class EigenFaces():
    def __init__(self, num_components):
        """
        Constructor
        """
        self._num_components = num_components;
        self._projections = []

    def train(self, images, labels):
        """
        Train eigenfaces
        """
        print "Train...",
        #copy labels
        self._labels = labels;

        #transform the numpe vector to shogun structure
        features = RealFeatures(images)
        #PCA
        self.pca = PCA()
        #set dimension
        self.pca.set_target_dim(self._num_components);
        #compute PCA
        self.pca.init(features)

        for sampleIdx in range(features.get_num_vectors()):
            v = features.get_feature_vector(sampleIdx);
            p = self.pca.apply_to_feature_vector(v);
            self._projections.insert(sampleIdx, p);

        print "ok!"

    def predict(self, image):
        """
        Predict the face
        """
        #image as row
        imageAsRow = np.asarray(image.reshape(image.shape[0]*image.shape[1],1),
                                np.float64);
        #project inthe subspace
        p = self.pca.apply_to_feature_vector(RealFeatures(imageAsRow).get_feature_vector(0));

        #min value to find the face
        minDist =1e100;
        #class
        minClass = -1;
        #search which face is the best match
        for sampleIdx in range(len(self._projections)):
            test = RealFeatures(np.asmatrix(p,np.float64).T)
            projection = RealFeatures(np.asmatrix(self._projections[sampleIdx],
                                        np.float64).T)
            dist = EuclideanDistance( test, projection).distance(0,0)

            if(dist < minDist ):
                minDist = dist;
                minClass = self._labels[sampleIdx];

        return minClass

    def getMean(self):
        """
        Return the mean vector
        """
        return self.pca.get_mean()

    def getEigenValues(self):
        """
        Return the eigenvalues vector
        """
        return self.pca.get_eigenvalues();

def readImages(list_filenames):
    """
    Read all the image. Image as rows
    """
    print "Reading images ...",
    #reserve space for the matrix
    images = np.empty( (IMAGE_HEIGHT*IMAGE_WIDHT, (len(list_filenames))-1))
    index = 0;
    for im_filename in list_filenames:
        #read image with opencv
        imagen= cv2.imread(im_filename, cv2.IMREAD_GRAYSCALE)
        #resize image -> problem with PCA N>>D
        imagen = cv2.resize(imagen, (IMAGE_HEIGHT, IMAGE_WIDHT),
                            interpolation=cv2.INTER_LINEAR);
        images[:,index] = imagen.reshape(imagen.shape[0]*imagen.shape[1],1).T;
        index=index + 1
        #don't read the last value (last value is to test eigenfaces)
        if( (len(list_filenames)-1)==index):
            break
    print "OK! "
    return images

#contains images (path) and labels
#DATABASE: AT&T Facedatabase
def get_imlist(path, NUM_PERSONS, NUM_IMAGES_PER_PERSON):

    """ Returns a list of filenames for NUM_PERSONS and NUM_IMAGES_PER_PERSON """
    list_filenames=[]
    list_labels=[]
    #add labels and images
    for num_person in range(NUM_PERSONS):
        for num_faces in range(NUM_IMAGES_PER_PERSON):
            filename =path+os.sep+str( (num_faces+1)+(num_person*10) )+'.pgm'
            #exits?
            if os.path.exists(filename):
                list_filenames.append(filename)
                list_labels.append(num_person)
    return [list_filenames, list_labels]

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())

if __name__ == '__main__':
    #return list of filenames and labels
    [list_filenames, list_labels] = get_imlist(os.pardir + os.sep +
                                         os.pardir + os.sep + os.pardir+
                                         os.sep + os.pardir + os.sep +'data' +
                                         os.sep +'att_dataset' + os.sep +
                                         'training', 25, 10)
    #read all images
    images = readImages(list_filenames);

    #subset of data image
    random.seed(0)
    subset = random.permutation(images.shape[1])
    test_images = images[:, subset[:N_SUBSET] ]
    show_images=[]
    show_images_titles= [];

    for n in range(N_SUBSET):
        show_images_titles.append("Sample image " + str(n));
        show_images.append(test_images[:,n].reshape(IMAGE_HEIGHT, IMAGE_WIDHT))

    #this class resolves the eigenfaces
    eigenfaces = EigenFaces(100)

    #train eigenfaces
    eigenfaces.train(images, list_labels)

    #test eigenfaces
    image = cv2.resize(cv2.imread(list_filenames[-1], cv2.IMREAD_GRAYSCALE),
                         (IMAGE_HEIGHT, IMAGE_WIDHT),  interpolation=cv2.INTER_LINEAR);
    print "predicted: ", eigenfaces.predict(image), " // real: " ,list_labels[-1]

    #Mean face
    #get mean and reshape ( height and width original size)
    mean = eigenfaces.getMean().reshape(IMAGE_HEIGHT, IMAGE_WIDHT);
    show_images_titles.append( "Mean face");
    show_images.append(mean)

    #Reconstruction with diferents values of eigenvectos

    #Read the last image of the file to test Eigenfaces
    image = cv2.resize(cv2.imread(list_filenames[0], cv2.IMREAD_GRAYSCALE),
                                 (IMAGE_HEIGHT, IMAGE_WIDHT), interpolation=cv2.INTER_LINEAR);
    #image as row
    imageAsRow = np.asarray(image.reshape(image.shape[0]*image.shape[1],1),
                            np.float64);

    reconstructions = range(10, 250, 50)
    reconstructions_images = np.empty( (IMAGE_HEIGHT, IMAGE_WIDHT*len(reconstructions) ), np.uint8)


    #Reconstruct 10 eigen vectors to 300, step 15
    for i in reconstructions:

        print "Reconstruct with " + str(i) + " eigenvectors"

        pca = PCA()
        #set dimension
        pca.set_target_dim(i);
        #compute PCA
        pca.init(RealFeatures(images))

        pca.apply_to_feature_vector(RealFeatures(imageAsRow)
                                    .get_feature_vector(0));

        #reconstruct
        projection = pca.apply_to_feature_vector(RealFeatures(imageAsRow)
                                                .get_feature_vector(0));

        reconstruction = np.asmatrix( np.asarray(projection, np.float64))* \
                         np.asmatrix( pca.get_transformation_matrix()).T
        reconstruction = reconstruction + pca.get_mean()

        #prepare the data to visualize in one window
        show_images_titles.append( str(i) + " eigenvectors" );
        show_images.append(reconstruction.reshape(IMAGE_HEIGHT, IMAGE_WIDHT))

    plot_gallery(show_images, show_images_titles, IMAGE_HEIGHT,
                 IMAGE_WIDHT, 4, 4);
    pl.show()
