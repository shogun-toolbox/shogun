#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

#include<shogun/base/init.h>
#include<shogun/features/DenseFeatures.h>
#include<shogun/lib/SGMatrix.h>
#include<shogun/preprocessor/PCA.h>
#include<shogun/preprocessor/FisherLDA.h>
#include<shogun/distance/EuclideanDistance.h>
#include<shogun/lib/OpenCV/CV2SGFactory.h>
#include<shogun/labels/MulticlassLabels.h>

#include<iostream>
#include<sstream>
#include<fstream>

using namespace cv;
using namespace std;
using namespace shogun;

#define NO_OF_EIGENFACES 359
#define NO_OF_FISHERFACES 39

static void read_csv(const string& filename, vector<Mat>& images,
		vector<float64_t>& labels, char separator=';')
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
	SGMatrix<float64_t> Ab1_f(AB.matrix, 1,AB.num_rows*AB.num_cols, false);
	return Ab1_f;
}

SGMatrix<float64_t> reverse_flatten(float64_t* matrix, int32_t rows, int32_t cols )
{
	SGMatrix<float64_t> Ab2_f(matrix, rows, cols, false);
	return Ab2_f;
}

int main()
{
	init_shogun_with_defaults();

	vector<Mat> images; // for containing the training images.
	vector<float64_t> labels; // for containing the labels for each of the images.
	vector<SGMatrix<float64_t> > eigenfaces; // for containing the eigenfaces.
                                             //will be used later in the code.
	// read the images & labels for each image from the text file.
	//read_csv("check_content1.txt", images, labels);
	read_csv("../../../data/att_dataset/training/content.txt", images, labels);

	// The dataset we are using has all the images of a constant dimension.
	const int height=images[0].rows; // height of each image.
	const int width=images[0].cols;  // width of each image.
	const int size=images.size();    // total number of images in the training dataset.
	const double length=width*height;// total features if each image is flattened.

	// A Shogun Matrix for holding all the training images in their flattened form.
	// no. of rows=total features per image.
	// np. of cols=total number of training images.
	SGMatrix<float64_t> Stacked_mats=SGMatrix<float64_t>(length, size);

	// In the following snippet we perform these three steps.
	// 1. convert each of the training images from OpenCV's Mat to Shogun's SGMatrix.
	// 2. flatten each of them.
	// 3. fill in them in the Stacked_mats defined above.

	SGMatrix<float64_t> temp1, temp2;
	for (int32_t j=0; j<size; ++j)
	{
		temp1=CV2SGFactory::get_sgmatrix<float64_t>(images[j]);
		temp2=flatten(temp1);
		for(int32_t i=0; i<length; ++i)
			Stacked_mats(i,j)=temp2(0,i);
	}

	// Now each column of Stacked_mats holds a single training image in its
	// flattened form like shown below.
	//	 [
	//	 [ Img1[0]   Img2[0]   Img3[0]   Img4[0] ......   ImgS[0];]
	//	 [ Img1[1]   Img2[1]   Img3[1]   Img4[1] ......   ImgS[1];]
	//	 [ Img1[2]   Img2[2]   Img3[2]   Img4[2] ......   ImgS[2];]
	//	 [   .          .        .          .    ......         .;]
	//	 [   .          .        .          .    ......         .;]
	//	 [ Img1[L]   Img2[L]   Img3[L]   Img4[L] ......   ImgS[L];]here L = length  and S = size.
	//	 ]

	// convert the Stacked_mats into the CDenseFeatures of Shogun.
	// From here on we will be performing our PCA algo in Shogun.
	CDenseFeatures<float64_t>* Face_features=new CDenseFeatures<float64_t>(Stacked_mats);
	SG_REF(Face_features)

	// We initialise the Preprocessor CPCA of Shogun which performs principal
	// component analysis on input features.
	CPCA* pca=new CPCA();
	SG_REF(pca);
	// Set the number of EigenFaces you want to use.
	pca->set_target_dim(NO_OF_EIGENFACES);

	// Run it on the Face_features defined above.
	pca->init(Face_features);

	// Get the mean of all the flattened training images. we will be using this
	// to centralize our test image later.
	SGVector<float64_t> mean=pca->get_mean();

	// Get the transformation matrix. It is a matrix which Shogun stores internally
	// whose dimensions are:  ( length * NO_OF_EIGENFACES )
	SGMatrix<float64_t> pca_eigenvectors=pca->get_transformation_matrix();

	// The following is the output of PCA. It's dimensions are:
	// ( NO_OF_EIGENFACES * size )
	SGMatrix<float64_t> pca_projection=pca->apply_to_feature_matrix(Face_features);

	// The overall summary of what we did above for say,
	// number of training images = 300 ( with each image of size 20*30 pixels)
	// & no. of eigenfaces required = 50 is,
	//
	//			Stacked_mats (10352 X 399)
	//			        *
	//			        |
	//			        |
	//			        *
	//			pca_projection (359  X 399)
	//
	// So in effect we just reduced the dimensions of each of our
	// training images. We will further reduce it using LDA.

	// Convert the Labels in the form of CMulticlassLabels
	SGVector<float64_t> labels_vector(size);
	for (int i=0;i<size;i++)
		labels_vector[i]=labels[i];
	CMulticlassLabels* actual_labels=new CMulticlassLabels(labels_vector);
	SG_REF(actual_labels);

	CDenseFeatures<float64_t>* pca_dense_feat=
		new CDenseFeatures<float64_t>(pca_projection);
	SG_REF(pca_dense_feat);

	// Applying the classical Fisher LDA algorithm.
	CFisherLDA fisherlda(CLASSIC_FLDA);

	// Unlike the PCA, we need to supply the labels also as the
	// projection done by LDA is class specific.
	fisherlda.init(pca_dense_feat, actual_labels, NO_OF_FISHERFACES);

	// Get the EigenVectors.
	SGMatrix<float64_t> lda_eigenvectors=
		fisherlda.get_transformation_matrix();

	// Get the LDA subspace Projection. This projection will be used against
	// the test samples for classification.
	SGMatrix<float64_t> lda_projection=
		fisherlda.apply_to_feature_matrix(pca_dense_feat);

	//		 This is what we have now
	//
	//		Stacked_mats (10352 X 399)
	//		        *
	//		        |
	//		        |
	//		        *
	//		pca_projection (359 X 399)
	//		        *
	//		        |
	//		        |
	//		        *
	//		lda_projection (39 X 399)

	// WFINAL=WPCA*WLDA
	SGMatrix<float64_t>Wfinal=SGMatrix<float64_t>::matrix_multiply
		(pca_eigenvectors, lda_eigenvectors);

	// Now we must get our test image readied. We simply follow the
	// steps that we did above for the training images.
	Mat testimage=imread("../../../data/att_dataset/testing/383.pgm",0);

	// we flatten the test image
	SGMatrix<float64_t> testimage_sgmat=CV2SGFactory::get_sgmatrix<float64_t>
		(testimage);
	temp2=flatten(testimage_sgmat); // temp2 is a column matrix.

	// we centralize the test image by subtracting the mean from it.
	SGVector<float64_t> testimage_sgvec(temp2.get_column_vector(0),
			temp2.num_cols, false);
	mean.scale_vector(-1, mean.vector, mean.vlen);
	add<linalg::Backend::NATIVE>(testimage_sgvec, mean, testimage_sgvec);

	// now we must project it into the PCA subspace. This is done by performing
	// the Dot product between testimage and the WFINAL.
	SGVector<float64_t> testimage_projected_vec(NO_OF_FISHERFACES);

	for (int i=0; i<NO_OF_FISHERFACES; ++i)
		testimage_projected_vec[i]=SGVector<float64_t>::dot(testimage_sgvec.vector,
		Wfinal.get_column_vector(i), testimage_sgvec.vlen);

	// For Eucledian Distance measure, we need to compare the above formed matrix
	// to the lda_projection
	// The one that gives the minimum distance between them will be identified
	// as the closest of the training image and that person will be identified
	// from the label of the identified image.

	// we need to have the Densefeature pointer of the lda_projection.
	// It is the lhs.
	CDenseFeatures<float64_t>* lhs=
		new CDenseFeatures<float64_t>(lda_projection);

	// and similarly we just need to convert the testimage_sgvec into the
	// DenseFeature pointer for the rhs.
	SGMatrix<float64_t>data_matrix(testimage_projected_vec.vlen, 1);
	CDenseFeatures<float64_t>* rhs=
		new CDenseFeatures<float64_t>(data_matrix);
	rhs->set_feature_vector(testimage_projected_vec,0);

	CEuclideanDistance* euclid=new CEuclideanDistance(lhs, rhs);
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

	SG_UNREF(euclid);
	SG_UNREF(actual_labels);
	SG_UNREF(pca_dense_feat);
	SG_UNREF(pca);
	SG_UNREF(Face_features);

}
