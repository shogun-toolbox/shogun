/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2010 Soeren Sonnenburg, Alexander Binder
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max Planck Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/preprocessor/RandomFourierGaussPreproc.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>

#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <iostream>
#include <algorithm>
#include <ctime>

//g++ -Wall -O3 -o tester preprocessor_randomfouriergauss.cpp -I ~/installed_software/shogun_git_localmaster/include/  -L  ~/installed_software/shogun_git_localmaster/lib/ -l shogun

using namespace shogun;


void gen_rand_data(float64_t* & feat, float64_t* & lab,const int32_t num,const int32_t dims,const float64_t dist)
{
	lab=SG_MALLOC(float64_t, num);
	feat=SG_MALLOC(float64_t, num*dims);

	for (int32_t i=0; i<num; i++)
	{
		if (i<num/2)
		{
			lab[i]=-1.0;

			for (int32_t j=0; j<dims; j++)
				feat[i*dims+j]=CMath::random(0.0,1.0)+dist;
		}
		else
		{
			lab[i]=1.0;

			for (int32_t j=0; j<dims; j++)
				feat[i*dims+j]=CMath::random(0.0,1.0)-dist;
		}
	}

}

int main()
{


	time_t a,b;

	// data generation parameters
	int32_t dims=6;
	float64_t dist=0.5;

	int32_t numtr=300;


	// random fourier transformation parameters
	int32_t randomfourier_featurespace_dim=900; // the typical application of the below preprocessor are cases with high input dimensionalities of some thousands
	const float64_t rbf_width=60; // gaussian kernel width
	// important trick for RFgauss to work: kernel width is set such that average inner kernel distance is close one
	// the rfgauss approximation breaks down if average inner kernel distances (~~ kernel width to small compared to variance of data) are too large
	// try rbf_width=0.1 to see how it fails! - you will see the problem in the large number of negative kernel entries (numnegratio) for the rfgauss linear kernel




	const int32_t kernel_cache=0;




	

	init_shogun();


	float64_t* feattr1(NULL);
	float64_t* labtr(NULL);

	a=time(NULL);
	std::cout << "generating train data"<<std::endl;
	gen_rand_data(feattr1,labtr,numtr,dims,dist);
	float64_t* feattr2=SG_MALLOC(float64_t, numtr*dims);
	std::copy(feattr1,feattr1+numtr*dims,feattr2);
	float64_t* feattr3=SG_MALLOC(float64_t, numtr*dims);
	std::copy(feattr1,feattr1+numtr*dims,feattr3);
	std::cout << "finished"<<std::endl;
	b=time(NULL);
	std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	// create train features

	std::cout << "initializing shogun train feature"<<std::endl;

	CDenseFeatures<float64_t>* featurestr1 = new CDenseFeatures<float64_t>(feattr1, dims, numtr);
	SG_REF(featurestr1);

	std::cout << "finished"<<std::endl;


	// create gaussian kernel
	std::cout << "computing gaussian train kernel"<<std::endl;

	CGaussianKernel* kerneltr1 = new CGaussianKernel(kernel_cache, rbf_width);
	SG_REF(kerneltr1);
	kerneltr1->init(featurestr1, featurestr1);

	
	std::cout << "finished"<<std::endl;

 // ***************************************
// now WITH the preprocessor

	std::cout << "initializing preprocessor"<<std::endl;

	CRandomFourierGaussPreproc *rfgauss=new CRandomFourierGaussPreproc;
	SG_REF(rfgauss);

	//rfgauss->get_io()->set_loglevel(MSG_DEBUG);

	// ************************************************************
	// set parameters of the preprocessor
	// ******************************** !!!!!!!!!!!!!!!!! CMath::sqrt(rbf_width/2.0)


	rfgauss->set_parameters(dims,randomfourier_featurespace_dim,CMath::sqrt(rbf_width/2.0));
	//**********8
	// initializing coefficients, do that one time and reuse coefficients for test feature transformation by: store coefficients using get_random_coefficients(...)  and set them for test features using set_random_coefficients(...)
	//*****************
	rfgauss->init_randomcoefficients_from_scratch();
	std::cout << "finished"<<std::endl;

	// create train features

	std::cout << "initializing shogun train feature again"<<std::endl;

	CDenseFeatures<float64_t>* featurestr2pre = new CDenseFeatures<float64_t>(feattr2, dims, numtr);
	SG_REF(featurestr2pre);


	std::cout << "finished"<<std::endl;
	//b=time(NULL);
	//std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	// ************************************************************
	// use preprocessor
	// **************************************************************

	// apply preprocessor
	a=time(NULL);
	std::cout << "applying preprocessor to train feature"<<std::endl;

	CDenseFeatures<float64_t>* featurestr2=rfgauss->apply_to_dotfeatures_sparse_or_dense_with_real(featurestr2pre);

	std::cout << "finished"<<std::endl;
	b=time(NULL);
	std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	// create linear kernel
	std::cout << "computing linear train kernel over preprocessed features"<<std::endl;

	CLinearKernel* kerneltr2 = new CLinearKernel();
	SG_REF(kerneltr2);
	kerneltr2->init(featurestr2, featurestr2);

	std::cout << "finished"<<std::endl;
	// save random coefficients and state data of preprocessor for use with a new preprocessor object (see lines following "// now the same with a new preprocessor to show the usage of set_randomcoefficients"
	// Alternative: use built-in serialization to load and save state data from/to a file!!!
 
	float64_t *randomcoeff_additive2, * randomcoeff_multiplicative2;
	int32_t dim_feature_space2,dim_input_space2;
	float64_t kernelwidth2;

	rfgauss->get_randomcoefficients(&randomcoeff_additive2,
				&randomcoeff_multiplicative2,
				&dim_feature_space2, &dim_input_space2, &kernelwidth2);




	// ************************************************************
	// use preprocessor with setting of rf coefficients (usually one would use that when computing testing features)
	// **************************************************************

	CDenseFeatures<float64_t>* featurestr3pre = new CDenseFeatures<float64_t>(feattr3, dims, numtr);
	SG_REF(featurestr3pre);


	CRandomFourierGaussPreproc *rfgauss2=new CRandomFourierGaussPreproc;
	SG_REF(rfgauss2);

	//rfgauss2->get_io()->set_loglevel(MSG_DEBUG);

	rfgauss2->set_parameters(dims,randomfourier_featurespace_dim,CMath::sqrt(rbf_width/2.0));
	rfgauss2->set_randomcoefficients(randomcoeff_additive2,
				randomcoeff_multiplicative2,
				dim_feature_space2, dim_input_space2, kernelwidth2);


	// apply preprocessor
	a=time(NULL);
	std::cout << "applying same preprocessor to test feature"<<std::endl;


	CDenseFeatures<float64_t>* featurestr3=rfgauss->apply_to_dotfeatures_sparse_or_dense_with_real(featurestr3pre);

	std::cout << "finished"<<std::endl;
	b=time(NULL);
	std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	//std::cout << "computing linear test kernel over preprocessed features"<<std::endl;

	CLinearKernel* kerneltr3 = new CLinearKernel();
	SG_REF(kerneltr3);
	kerneltr3->init(featurestr3, featurestr3);
	//std::cout << "finished"<<std::endl;
	//b=time(NULL);
	//std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	
a=time(NULL);
std::cout << "computing effective kernel widths (means of inner distances)"<<std::endl;

int32_t m= kerneltr1->get_num_vec_lhs(), n= kerneltr1->get_num_vec_rhs();

SGMatrix<float64_t> kertr1= kerneltr1->get_kernel_matrix ();

std::cout << "kernel size "<< m << " "<< n <<std::endl;

float64_t avgdist1=0;
for(int i=0; i<m ;++i)
{
	for(int l=0; l<i ;++l)
	{
		avgdist1+= -CMath::log(kertr1[i+l*m])*2.0/m/(m+1.0);
	}
}

SGMatrix<float64_t> kertr2= kerneltr2->get_kernel_matrix ();
SGMatrix<float64_t> kertr3= kerneltr3->get_kernel_matrix ();

float64_t diffs12=0;
float64_t diffs23=0;

float64_t avgdist2=0;
float64_t numnegratio=0;
for(int i=0; i<m ;++i)
{
	for(int l=0; l<i ;++l)
	{
		if(kertr2[i+l*m]<=0)
		{
			numnegratio+=2.0/m/(m+1.0);
		}
		else
		{
		avgdist2+= -CMath::log(std::max(kertr2[i+l*m],1e-10))*2.0/m/(m+1.0);
		}

		diffs12+=fabs(kertr1[i+l*m]-kertr2[i+l*m])/kertr1[i+l*m]*2.0/m/(m+1.0);
		diffs23+=fabs(kertr2[i+l*m]-kertr3[i+l*m])*2.0/m/(m+1.0);
	}
}






std::cout << " absolute differences between rf approximation with inited coefficients and saved and set coefficients ( get_randomcoefficients(...)+set_randomcoefficients(...) " << diffs23  <<std::endl;

std::cout << "finished"<<std::endl;
b=time(NULL);
std::cout<< "elapsed time in seconds "<<b-a <<std::endl;
std::cout <<std::endl<< "effective kernel widths for gaussian kernel versus RFgauss approximation "<< avgdist1 << " " <<avgdist2/(1.0-numnegratio) << std::endl<< " ratio of negative entries in RFgauss approx kernel (that are bad results) "<< numnegratio<<std::endl;



std::cout <<std::endl<<std::endl<< " RELATIVE DIFFERENCES between true gaussian kernel and rf approximation " << diffs12  <<std::endl<<std::endl<<std::endl;





















	SG_FREE(randomcoeff_additive2);
	SG_FREE(randomcoeff_multiplicative2);

	SG_FREE(labtr);

	//SG_FREE(kertr1);
	//SG_FREE(kertr2);
	//SG_FREE(kertr3);


	//SG_UNREF(kerneltr1);
	//SG_UNREF(kerneltr2);
	//SG_UNREF(kerneltr3);

	SG_UNREF(featurestr1);
	SG_UNREF(featurestr2);
	SG_UNREF(featurestr3);


	SG_UNREF(rfgauss);
	SG_UNREF(rfgauss2);
	exit_shogun();
	return 0;
}
