#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

#include<shogun/base/init.h>
#include<shogun/features/DenseFeatures.h>
#include<shogun/lib/SGMatrix.h>
#include<shogun/preprocessor/PCA.h>
#include<shogun/distance/EuclideanDistance.h>
#include<shogun/lib/OpenCV/CV2SGFactory.h>

#include<iostream>
#include<sstream>
#include<fstream>

using namespace cv;
using namespace std;
using namespace shogun;

#define NO_OF_EIGENFACES 50

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator=';')
{
    std::ifstream file(filename.c_str(), ifstream::in);
    string line, path, classlabel;
    while (getline(file, line))
    {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty())
        {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

SGMatrix<float64_t> flatten(SGMatrix<float64_t> AB)
{
    SGMatrix<float64_t> flat(AB.matrix, 1,AB.num_rows*AB.num_cols, false);
    return flat;
}

SGMatrix<float64_t> reverse_flatten(float64_t* matrix, int32_t rows, int32_t cols )
{
    SGMatrix<float64_t> r_flat(matrix, rows, cols, false);
    return r_flat;
}





int main()
{
    init_shogun_with_defaults();

    vector<Mat> images; // for containing the training images.
    vector<int> labels; // for containing the labels for each of the images.
    vector<SGMatrix<float64_t> > eigenfaces; // for containing the eigenfaces for the images. will be used later in the code.


    // read the images & labels for each image from the text file.
    read_csv("../../../data/att_dataset/training/content.txt", images, labels);


    // The dataset we are using has all the images of a constant dimension.
    const int height=images[0].rows; // height of each image.
    const int width=images[0].cols; // width of each image.
    const int size=images.size(); // total number of images in the training dataset.
    const double length=width*height; // total features if each image is flattened.


    // A Shogun Matrix for holding all the training images in their flattened form.
    // no. of rows=total features per image.
    // np. of cols=total number of training images.
    SGMatrix<float64_t> Stacked_mats=SGMatrix<float64_t>(length, size);   


    // In the following snippet we perform these three steps.
    // 1. convert each of the training images from OpenCV's Mat to Shogun's SGMatrix.
    // 2. flatten each of them.
    // 3. fill in them in the Stacked_mats defined above.

    SGMatrix<float64_t> temp1, temp2;
    for (int32_t j=0; j< size; ++j)
    {
        temp1=CV2SGFactory::get_sgmatrix<float64_t>(images[j]);
        temp2=flatten(temp1);
        for(int32_t i=0; i<length; ++i)
        {
            Stacked_mats(i,j)=temp2(0,i);
        }
    }

    // Now each column holds a single training image in its flattened form like shown below.
    //   [
    //   [ Img1[0]   Img2[0]   Img3[0]   Img4[0] ......   ImgS[0];]
    //   [ Img1[1]   Img2[1]   Img3[1]   Img4[1] ......   ImgS[1];]
    //   [ Img1[2]   Img2[2]   Img3[2]   Img4[2] ......   ImgS[2];]
    //   [   .          .        .          .    ......      .   ;]
    //   [   .          .        .          .    ......      .   ;]
    //   [ Img1[L]   Img2[L]   Img3[L]   Img4[L] ......   ImgS[L];]     here L=length  and S=size.
    //                                                            ]


    // convert the Stacked_mats into the CDenseFeatures of Shogun. 
    // From here on we will be performing our PCA algo in Shogun. 
    CDenseFeatures<float64_t>* Face_features=new CDenseFeatures<float64_t>(Stacked_mats); 
    SG_REF(Face_features); 
    
    // We initialise the Preprocessor CPCA of Shogun which performs principal 
    // component analysis on input features.
    CPCA* pca=new CPCA();
    SG_REF(pca);

    // Set the number of EigenFaces you want to use.
    pca->set_target_dim( NO_OF_EIGENFACES );

    // Run it on the Face_features defined above.
    pca->init(Face_features);

    // Get the mean of all the flattened training images. we will be using this
    // to centralize our test image later.
    SGVector<float64_t> mean=pca->get_mean() ;  

    // Get the transformation matrix. It is a matrix which Shogun stores internally 
    // whose dimensions are:  ( length * NO_OF_EIGENFACES )     
    SGMatrix<float64_t> transmat=pca->get_transformation_matrix();

    // The following is the output of PCA. It's dimensions are: ( NO_OF_EIGENFACES * size )
	SGMatrix<float64_t> finalmat=pca->apply_to_feature_matrix(Face_features);
	
    // The overall summary of what we did above for say,
    // number of training images=300 ( with each image of size 20*30 pixels)
    // & no. of eigenfaces required=50 is,
    //
    //                                      Stacked_mats (600 X 300)
    //                                          *
    //                                          |
    //                                          |
    //                                          *               
    //                                       finalmat    (50  X 300)
    //
    // So in effect we just reduced the dimensions of each of our training images
    // by 12 times.
    // Just this is the work of PCA in Eigenfaces face recognition algorithm.




    // Now we must get our test image readied. We simply follow the steps that we did above for the training images.
    Mat testimage=imread("../../../data/att_dataset/testing/383.pgm",0);




    // we flatten the test image
    SGMatrix<float64_t> testimage_sgmat=CV2SGFactory::get_sgmatrix<float64_t>(testimage);
    temp2=flatten(testimage_sgmat); // temp2 is a column matrix.




    // we centralize the test image by subtracting the mean from it.
    SGVector<float64_t> testimage_sgvec( temp2.get_column_vector(0), temp2.num_cols, false); 
    mean.scale_vector(-1, mean.vector, mean.vlen);


    testimage_sgvec.add(mean);




    // now we must project it into the PCA subspace. This is done by performing 
    // the Dot product between testimage and the transformation matrix. 
    float64_t testimage_projected_array[NO_OF_EIGENFACES];
    for (int i=0; i<NO_OF_EIGENFACES; ++i)
    {
        testimage_projected_array[i]=
        SGVector<float64_t>::dot(testimage_sgvec.vector, transmat.get_column_vector(i), testimage_sgvec.vlen); 
    }

    // we here convert the projected testimage(array) into Shogun Vector. 
    SGVector<float64_t> testimage_projected_vec(testimage_projected_array, NO_OF_EIGENFACES, false);




    // For Eucledian Distance measure, we need to compare the above formed matrix
    // to the finalmat. 
    // The one that gives the minimum distance between them will be identified 
    // as the closest of the training image and that person will be identified 
    // from the label of the identified image. 

    // we need to have the Densefeature pointer of the finalmat.
    // finalmat_densefeature_ptr is the lhs.
    CDenseFeatures<float64_t>* finalmat_densefeature_ptr=new CDenseFeatures<float64_t>(finalmat);
    SG_REF(finalmat_densefeature_ptr); 

    // and similarly we just need to convert the testimage_sgvec into the DenseFeature pointer for the rhs.
    SGMatrix<float64_t>data_matrix(testimage_projected_vec.vlen, 1);

    CDenseFeatures<float64_t>* testimage_dense=new CDenseFeatures<float64_t>(data_matrix);
    SG_REF(testimage_dense);
    testimage_dense->set_feature_vector(testimage_projected_vec,0);



    CEuclideanDistance* euclid=new CEuclideanDistance(finalmat_densefeature_ptr,testimage_dense);   
    SG_REF(euclid);
    float64_t distance_array[size];
    int min_index=0;
    for (int i=0; i<size; ++i)
    {
        distance_array[i]=euclid->distance(i,0);
        if(distance_array[i]<distance_array[min_index])
        min_index=i;
    }
    cout<<"index:"<<min_index<<endl;
    cout<<"label:"<<labels[min_index]<<endl;


    SG_UNREF(finalmat_densefeature_ptr);
    SG_UNREF(Face_features);
    SG_UNREF(euclid);
    SG_UNREF(testimage_dense);
    SG_UNREF(pca);
return 0;
}
