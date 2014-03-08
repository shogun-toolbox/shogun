#!/usr/bin/env python

# This example is about learning and using eigenfaces in Shogun. We demonstrate how to use them for a set of faces.

import cv2
import numpy as np
from modshogun import RealFeatures
from modshogun import PCA
from numpy import linalg as LA
import math

IMAGE_WIDHT = 40
IMAGE_HEIGHT = 40

class EigenFaces():
    def __init__(self, num_components):
        """
        Constructor
        """
        self._num_components = num_components;
        self._threshold = 1e100;
        self._projections = []

    def subspaceProject(self, _w, mean, v):
        """
        Project in the subspace
        """
        X = v - mean;
        Y = np.asmatrix(X)* np.asmatrix(_w).T;
        return Y

    def subspaceReconstruct(self, _w, mean, v):
        """
        subspace reconstruct
        """
        X = np.asmatrix(v)*np.asmatrix(_w);
        X = X + mean;
        return X



    def train(self, images, labels):
        """
        Train eigenfaces
        """
        print "Train..."
        #copy labels
        self._labels = labels;

        #transform the numpe vector to shogun structure
        features = RealFeatures(images)

        #PCA
        preprocessor = PCA()
        # set dimension
        preprocessor.set_target_dim(self._num_components);
        # compute PCA     
        preprocessor.init(features)
        # get eigen vector and sort the matrix
        eigenvectors = preprocessor.get_transformation_matrix();     
        eigenvectors =  eigenvectors.T.reshape(1, self._num_components*features.get_num_features());
        self.eigenvectors_mainComponents = np.empty( (self._num_components, (images.shape[0])))
        index = 0;
        for i in range (self.eigenvectors_mainComponents.shape[1]):
            for j in range(self.eigenvectors_mainComponents.shape[0]):
                self.eigenvectors_mainComponents[j, i] = eigenvectors[:, (index*self._num_components)+(self._num_components-1)-j ]
            index = index+1;
        # get eigenvector        
        self.eigenvalues = preprocessor.get_eigenvalues();   
        # get mean
        self.mean = preprocessor.get_mean();  
       
        for sampleIdx in range(features.get_num_vectors()):
            v = features.get_feature_vector(sampleIdx);
            p = self.subspaceProject(self.eigenvectors_mainComponents, self.mean, v);
            self._projections.insert(sampleIdx, p);

        print "Train ok!"

    def predict(self, image):
        """
        Predict the face
        """
        # image as row
        imageAsRow = image.reshape(image.shape[0]*image.shape[1],1).T;
        # project inthe subspace
        p = self.subspaceProject(self.eigenvectors_mainComponents, self.mean, imageAsRow);
           
        minDist =1e100; # min value to find the face
        minClass = -1;  #class      
        #search which face is the best match
        for sampleIdx in range(len(self._projections)):
            result=0;
            for i in range(p.shape[1]):
                result = result + (p[0, i]-self._projections[sampleIdx][0, i])**2;
            dist = math.sqrt(result)
            if(dist < minDist ):
                minDist = dist;
                minClass = self._labels[sampleIdx];

        return minClass

    def getMean(self):
        """
        Return the mean vector
        """
        return self.mean

    def getEigenVectors(self):
        """
        Return the eigenvectors vector
        """
        return self.eigenvectors_mainComponents

    def getEigenValues(self):
        """
        Return the eigenvalues vector
        """
        return self.eigenvalues

# file in csv format that contains the path to the images and the labels
# DATABASE: AT&T Facedatabase
FILE_NAME = "test_original.txt"

def read_csv(): 
    """
    Read the file, format
    """  
    print "Read CSV file ..." 
    # open file
    f = open(FILE_NAME,'r')
    # inicialize structures
    list_filenames=[]
    list_labels = []
    #read file
    for line in f.readlines():
        words = line.split(";")
        list_labels.append(int(words[1]))
        list_filenames.append(words[0])   
    print "OK!" 
    # return filenames and labels
    return [list_filenames, list_labels]

def readImages(list_filenames):
    """
    Read all the image. Image as rows
    """
    print "Reading images ..."
    # reserve space for the matrix
    images = np.empty( (IMAGE_HEIGHT*IMAGE_WIDHT, (len(list_filenames))-1))
    index = 0;
    for im_filename in list_filenames:
        # read image with opencv
        imagen= cv2.imread(im_filename, cv2.IMREAD_GRAYSCALE)
        # resize image -> problem with PCA N>>D
        imagen = cv2.resize(imagen, (IMAGE_HEIGHT, IMAGE_WIDHT));
        images[:,index] = imagen.reshape(imagen.shape[0]*imagen.shape[1],1).T;
        index=index + 1
        # don't read the last value (last value is to test eigenfaces)         
        if( (len(list_filenames)-1)==index):
            break
    print "OK! " 
    return images

if __name__ == '__main__':
    # read csv
    [list_filenames, list_labels] = read_csv()

    #read all images
    images = readImages(list_filenames);

    #  this class resolves the eigenfaces
    eigenfaces = EigenFaces(300)
    ############################
    # train eigenfaces
    ############################
    eigenfaces.train(images, list_labels)

    ############################
    # test eigenfaces
    ############################
    image = cv2.resize(cv2.imread(list_filenames[-1], cv2.IMREAD_GRAYSCALE), (IMAGE_HEIGHT, IMAGE_WIDHT));

    print "predicted: ", eigenfaces.predict(image), " / real: " ,list_labels[-1] 

    ###############
    #   Mean face
    ###############
    # get mean and reshape ( height and width original size)
    mean = eigenfaces.getMean().reshape(IMAGE_HEIGHT, IMAGE_WIDHT);
    # create mean normalize image (0, 255, type = uint8)   
    mean_normalize = np.zeros((IMAGE_HEIGHT, IMAGE_WIDHT,1), np.uint8)
    # normalize
    mean_normalize= cv2.normalize(mean, mean_normalize, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1);
    #show
    cv2.imshow("mean_normalize", mean_normalize)
    cv2.waitKey(0)


    ###################
    #   Reconstruction with diferents values of eigenvectos
    ###################
    # Read the last image of the file to test Eigenfaces
    image = cv2.resize(cv2.imread(list_filenames[0], cv2.IMREAD_GRAYSCALE), (IMAGE_HEIGHT, IMAGE_WIDHT));
    # image as row
    imageAsRow = image.reshape(image.shape[0]*image.shape[1],1).T;
    # eigenvector
    W = eigenfaces.getEigenVectors();
    mean = eigenfaces.getMean();
    # Reconstruct 10 eigen vectors to 300, step 15
    for i in range(10, 300, 15):
        # if _num_components if lower than dthe eigenvector to reconstruct -> break
        if W.shape[0]<i:
            break
        #reconstruct
        projection = eigenfaces.subspaceProject(W[0:i, :], mean, imageAsRow);
        reconstruction = eigenfaces.subspaceReconstruct(W[0:i, :], mean, projection);
        #normlize image
        reconstruction_normalize = np.zeros((IMAGE_HEIGHT, IMAGE_WIDHT,1), np.uint8)
        reconstruction_normalize = cv2.normalize(reconstruction, reconstruction_normalize, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1);
        reconstruction_normalize = reconstruction_normalize.reshape(IMAGE_HEIGHT, IMAGE_WIDHT)
        # show reconstruction        
        cv2.imshow("reconstruction",reconstruction_normalize)
        cv2.waitKey(0)


