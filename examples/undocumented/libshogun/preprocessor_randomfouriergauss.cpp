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
#include <shogun/preproc/RandomFourierGaussPreproc.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/lib/Mathematics.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>

#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <iostream>
#include <algorithm>
#include <ctime>

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
	CMath::display_vector(lab,num);
	CMath::display_matrix(feat,dims, num);
}

int main()
{

	time_t a,b;
	int32_t dims=6000;
	float64_t dist=0.5;

	int32_t randomfourier_featurespace_dim=500; // the typical application of the below preprocessor are cases with high input dimensionalities of some thousands

	int32_t numtr=3000;
	int32_t numte=3000;

	const int32_t feature_cache=0;
	const int32_t kernel_cache=0;

	// important trick for RFgauss to work: kernel width is set such that average inner kernel distance is close one
	// the rfgauss approximation breaks down if average inner kernel distances (~~ kernel width to small compared to variance of data) are too large
	// try rbf_width=0.1 to see how it fails! - you will see the problem in the large number of negative kernel entries (numnegratio) for the rfgauss linear kernel

	const float64_t rbf_width=4000;
	const float64_t svm_C=10;
	const float64_t svm_eps=0.001;

	init_shogun();


	float64_t* feattr(NULL);
	float64_t* labtr(NULL);

	a=time(NULL);
	std::cout << "generating train data"<<std::endl;
	gen_rand_data(feattr,labtr,numtr,dims,dist);
	float64_t* feattr2=SG_MALLOC(float64_t, numtr*dims);
	std::copy(feattr,feattr+numtr*dims,feattr2);
	std::cout << "finished"<<std::endl;
	b=time(NULL);
	std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	float64_t* featte(NULL);
	float64_t* labte(NULL);

	a=time(NULL);
	std::cout << "generating test data"<<std::endl;
	gen_rand_data(featte,labte,numte,dims,dist);
	float64_t* featte2=SG_MALLOC(float64_t, numtr*dims);
	std::copy(featte,featte+numtr*dims,featte2);
	float64_t* featte3=SG_MALLOC(float64_t, numtr*dims);
	std::copy(featte,featte+numtr*dims,featte3);
	std::cout << "finished"<<std::endl;
	b=time(NULL);
	std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	// create train labels
	CLabels* labelstr=new CLabels();
	labelstr->set_labels(labtr, numtr);
	SG_REF(labelstr);

	// create train features
	a=time(NULL);
	std::cout << "initializing shogun train feature"<<std::endl;

	CDenseFeatures<float64_t>* featurestr1 = new CDenseFeatures<float64_t>(feature_cache);
	SG_REF(featurestr1);


	featurestr1->set_feature_matrix(feattr, dims, numtr);
	std::cout << "finished"<<std::endl;
	//b=time(NULL);
	//std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	// create gaussian kernel
//	std::cout << "computing gaussian train kernel"<<std::endl;

	CGaussianKernel* kerneltr1 = new CGaussianKernel(kernel_cache, rbf_width);
	SG_REF(kerneltr1);
	kerneltr1->init(featurestr1, featurestr1);

	// create svm via libsvm and train
	CLibSVM* svm1 = new CLibSVM(svm_C, kerneltr1, labelstr);
	SG_REF(svm1);
	svm1->set_epsilon(svm_eps);

	a=time(NULL);
	std::cout << "training SVM over gaussian kernel"<<std::endl;
	svm1->train();
	std::cout << "finished"<<std::endl;
	b=time(NULL);
	std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	printf("num_sv:%d b:%f\n", svm1->get_num_support_vectors(), svm1->get_bias());

	a=time(NULL);
	std::cout << "initializing shogun test feature"<<std::endl;

	CDenseFeatures<float64_t>* featureste1 = new CDenseFeatures<float64_t>(feature_cache);
	SG_REF(featureste1);


	featureste1->set_feature_matrix(featte, dims, numte);
	std::cout << "finished"<<std::endl;
	//b=time(NULL);
	//std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	//std::cout << "computing gaussian test kernel"<<std::endl;
	CGaussianKernel* kernelte1 = new CGaussianKernel(kernel_cache, rbf_width);
	SG_REF(kernelte1);
	kernelte1->init(featurestr1, featureste1);
	svm1->set_kernel(kernelte1);

	a=time(NULL);
	std::cout << "scoring gaussian test kernel"<<std::endl;


	std::vector<float64_t> scoreste1(numte);
	float64_t err1=0;
	for(int32_t i=0; i< numte ;++i)
	{
		scoreste1[i]=svm1->classify_example(i);
		if(scoreste1[i]*labte[i]<0)
		{
			err1+=1.0/numte;
		}
	}

	std::cout << "finished"<<std::endl;
	b=time(NULL);
	std::cout<< "elapsed time in seconds "<<b-a <<std::endl;



 // ***************************************
// now WITH the preprocessor
	a=time(NULL);
	std::cout << "initializing preprocessor"<<std::endl;

	CRandomFourierGaussPreproc *rfgauss=new CRandomFourierGaussPreproc;
	SG_REF(rfgauss);

	rfgauss->get_io()->set_loglevel(MSG_DEBUG);

	// ************************************************************
	// set parameters of the preprocessor
	// ******************************** !!!!!!!!!!!!!!!!! CMath::sqrt(rbf_width/2.0)
	rfgauss->set_kernelwidth( CMath::sqrt(rbf_width/2.0) );
	rfgauss->set_dim_input_space(dims);
	rfgauss->set_dim_feature_space(randomfourier_featurespace_dim);

	std::cout << "finished"<<std::endl;
	//b=time(NULL);
	//std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	// create train features

	a=time(NULL);
	std::cout << "initializing shogun train feature again"<<std::endl;

	CDenseFeatures<float64_t>* featurestr2 = new CDenseFeatures<float64_t>(feature_cache);
	SG_REF(featurestr2);
	featurestr2->set_feature_matrix(feattr2, dims, numtr);

	std::cout << "finished"<<std::endl;
	//b=time(NULL);
	//std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	// ************************************************************
	// use preprocessor
	// **************************************************************
	// add preprocessor
	featurestr2->add_preproc(rfgauss);
	// apply preprocessor
	a=time(NULL);
	std::cout << "applying preprocessor to train feature"<<std::endl;

	featurestr2->apply_preproc();
	std::cout << "finished"<<std::endl;
	b=time(NULL);
	std::cout<< "elapsed time in seconds "<<b-a <<std::endl;


	// save random coefficients and state data of preprocessor for use with a new preprocessor object (see lines following "// now the same with a new preprocessor to show the usage of set_randomcoefficients"
	// Alternative: use built-in serialization to load and save state data from/to a file!!!

	float64_t *randomcoeff_additive2, * randomcoeff_multiplicative2;
	int32_t dim_feature_space2,dim_input_space2;
	float64_t kernelwidth2;

	rfgauss->get_randomcoefficients(&randomcoeff_additive2,
				&randomcoeff_multiplicative2,
				&dim_feature_space2, &dim_input_space2, &kernelwidth2);

	// create linear kernel
	//std::cout << "computing linear train kernel over preprocessed features"<<std::endl;

	CLinearKernel* kerneltr2 = new CLinearKernel();
	SG_REF(kerneltr2);
	kerneltr2->init(featurestr2, featurestr2);

	// create svm via libsvm and train
	CLibSVM* svm2 = new CLibSVM(svm_C, kerneltr2, labelstr);
	SG_REF(svm2);
	svm2->set_epsilon(svm_eps);
	a=time(NULL);
	std::cout << "training SVM over linear kernel over preprocessed features"<<std::endl;

	svm2->train();
	std::cout << "finished"<<std::endl;
	b=time(NULL);
	std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	printf("num_sv:%d b:%f\n", svm2->get_num_support_vectors(), svm2->get_bias());
	a=time(NULL);
	std::cout << "initializing shogun test feature again"<<std::endl;

	CDenseFeatures<float64_t>* featureste2 = new CDenseFeatures<float64_t>(feature_cache);
	SG_REF(featureste2);
	featureste2->set_feature_matrix(featte2, dims, numte);
	std::cout << "finished"<<std::endl;
	//b=time(NULL);
	//std::cout<< "elapsed time in seconds "<<b-a <<std::endl;


	// ************************************************************
	// use preprocessor
	// **************************************************************
	CRandomFourierGaussPreproc *rfgauss2=new CRandomFourierGaussPreproc;
	SG_REF(rfgauss2);

	rfgauss2->get_io()->set_loglevel(MSG_DEBUG);

	// add preprocessor
	featureste2->add_preproc(rfgauss);
	// apply preprocessor
	a=time(NULL);
	std::cout << "applying same preprocessor to test feature"<<std::endl;

	featureste2->apply_preproc();
	std::cout << "finished"<<std::endl;
	b=time(NULL);
	std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	//std::cout << "computing linear test kernel over preprocessed features"<<std::endl;

	CLinearKernel* kernelte2 = new CLinearKernel();
	SG_REF(kernelte2);
	kernelte2->init(featurestr2, featureste2);
	//std::cout << "finished"<<std::endl;
	//b=time(NULL);
	//std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	svm2->set_kernel(kernelte2);
	a=time(NULL);
	std::cout << "scoring linear test kernel over preprocessed features"<<std::endl;

	std::vector<float64_t> scoreste2(numte);

	float64_t err2=0;
	for(int32_t i=0; i< numte ;++i)
	{
		scoreste2[i]=svm2->classify_example(i);
		if(scoreste2[i]*labte[i]<0)
		{
			err2+=1.0/numte;
		}
	}
	std::cout << "finished"<<std::endl;
	b=time(NULL);
	std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	std::cout << "pausing 12 seconds"<<std::endl;
	sleep(12);
	// ************************************************************
	// compare results
	// **************************************************************
	int32_t num_labeldiffs=0;
	float64_t avg_scorediff=0;
	for(int32_t i=0; i< numte ;++i)
	{
		if( (int32_t)CMath::sign(scoreste1[i]) != (int32_t)CMath::sign(scoreste2[i]))
		{
			++num_labeldiffs;
		}
		avg_scorediff+=CMath::abs(scoreste1[i]-scoreste2[i])/numte;
		std::cout<< "at sample i"<< i <<" label 1= " << CMath::sign(scoreste1[i]) <<" label 2= " << CMath::sign(scoreste2[i])<< " scorediff " << scoreste1[i] << " - " <<scoreste2[i] <<" = " << CMath::abs(scoreste1[i]-scoreste2[i])<<std::endl;
	}
	std::cout << "usedwidth for rbf kernel"<<	kerneltr1->get_width() << " " <<	kernelte1->get_width()<<std::endl;

std::cout<< "number of different labels between gaussian kernel and rfgauss "<< num_labeldiffs<< " out of "<< numte << " labels "<<std::endl;
std::cout<< "average test sample SVM output score difference between gaussian kernel and rfgauss "<< avg_scorediff<<std::endl;
std::cout<< "classification errors gaussian kernel and rfgauss  "<< err1 << " " <<err2<<std::endl;

a=time(NULL);
std::cout << "computing effective kernel widths (means of inner distances)"<<std::endl;

int32_t m, n;
float64_t * kertr1;
kerneltr1->get_kernel_matrix ( &kertr1, &m, &n);

std::cout << "kernel size "<< m << " "<< n <<std::endl;

float64_t avgdist1=0;
for(int i=0; i<m ;++i)
{
	for(int l=0; l<i ;++l)
	{
		avgdist1+= -CMath::log(kertr1[i+l*m])*2.0/m/(m+1.0);
	}
}

float64_t * kertr2;
kerneltr2->get_kernel_matrix (&kertr2,&m, &n);


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
	}
}
std::cout << "finished"<<std::endl;
b=time(NULL);
std::cout<< "elapsed time in seconds "<<b-a <<std::endl;
std::cout << "effective kernel width for gaussian kernel and RFgauss "<< avgdist1 << " " <<avgdist2/(1.0-numnegratio) << std::endl<< " numnegratio (negative entries in RFgauss approx kernel)"<< numnegratio<<std::endl;





 // **********************************************
// now the same with a new preprocessor to show the usage of set_randomcoefficients
// ********************************************8

	CDenseFeatures<float64_t>* featureste3 = new CDenseFeatures<float64_t>(feature_cache);
	SG_REF(featureste3);
	featureste3->set_feature_matrix(featte3, dims, numte);
	std::cout << "finished"<<std::endl;
	//b=time(NULL);
	//std::cout<< "elapsed time in seconds "<<b-a <<std::endl;


	// ************************************************************
	// use preprocessor
	// **************************************************************
	rfgauss2->set_randomcoefficients(
		randomcoeff_additive2,
		randomcoeff_multiplicative2,
		dim_feature_space2, dim_input_space2, kernelwidth2);

	// add preprocessor
	featureste3->add_preproc(rfgauss2);
	// apply preprocessor
	a=time(NULL);
	std::cout << "applying same preprocessor to test feature"<<std::endl;

	featureste3->apply_preproc();
	std::cout << "finished"<<std::endl;
	b=time(NULL);
	std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	//std::cout << "computing linear test kernel over preprocessed features"<<std::endl;

	CLinearKernel* kernelte3 = new CLinearKernel();
	SG_REF(kernelte3);
	kernelte2->init(featurestr2, featureste3);
	//std::cout << "finished"<<std::endl;
	//b=time(NULL);
	//std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	svm2->set_kernel(kernelte3);
	a=time(NULL);
	std::cout << "scoring linear test kernel over preprocessed features"<<std::endl;

	std::vector<float64_t> scoreste3(numte);

	float64_t err3=0;
	for(int32_t i=0; i< numte ;++i)
	{
		scoreste3[i]=svm2->classify_example(i);
		if(scoreste3[i]*labte[i]<0)
		{
			err3+=1.0/numte;
		}
	}
	std::cout << "finished"<<std::endl;
	b=time(NULL);
	std::cout<< "elapsed time in seconds "<<b-a <<std::endl;

	std::cout << "pausing 12 seconds"<<std::endl;
	sleep(12);
	// ************************************************************
	// compare results
	// **************************************************************
	num_labeldiffs=0;
	avg_scorediff=0;
	for(int32_t i=0; i< numte ;++i)
	{
		if( (int32_t)CMath::sign(scoreste1[i]) != (int32_t)CMath::sign(scoreste3[i]))
		{
			++num_labeldiffs;
		}
		avg_scorediff+=CMath::abs(scoreste1[i]-scoreste3[i])/numte;
		std::cout<< "at sample i"<< i <<" label 1= " << CMath::sign(scoreste1[i]) <<" label 2= " << CMath::sign(scoreste3[i])<< " scorediff " << scoreste1[i] << " - " <<scoreste3[i] <<" = " << CMath::abs(scoreste1[i]-scoreste3[i])<<std::endl;
	}

std::cout<< "number of different labels between gaussian kernel and rfgauss "<< num_labeldiffs<< " out of "<< numte << " labels "<<std::endl;
std::cout<< "average test sample SVM output score difference between gaussian kernel and rfgauss "<< avg_scorediff<<std::endl;
std::cout<< "classification errors gaussian kernel and rfgauss  "<< err1 << " " <<err3<<std::endl;
























	SG_FREE(randomcoeff_additive2);
	SG_FREE(randomcoeff_multiplicative2);

	SG_FREE(labtr);
	SG_FREE(labte);
	SG_FREE(kertr1);
	SG_FREE(kertr2);

	SG_UNREF(labelstr);
	SG_UNREF(kerneltr1);
	SG_UNREF(kerneltr2);
	SG_UNREF(kernelte1);
	SG_UNREF(kernelte2);
	SG_UNREF(kernelte3);
	SG_UNREF(featurestr1);
	SG_UNREF(featurestr2);
	SG_UNREF(featureste1);
	SG_UNREF(featureste2);
	SG_UNREF(featureste3);
	SG_UNREF(svm1);
	SG_UNREF(svm2);
	SG_UNREF(rfgauss);
	SG_UNREF(rfgauss2);
	exit_shogun();
	return 0;
}
