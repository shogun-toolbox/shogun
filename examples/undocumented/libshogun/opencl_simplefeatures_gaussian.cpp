#define USE_OPENCL

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Soeren Sonnenburg
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST and Max Planck Society
 */
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>
#include <iostream>
#include <viennacl/ocl/utils.hpp>
#include <stdlib.h>
#include <stdio.h>
#include "Timer.hpp"


using namespace shogun;

#define NUM 10000
#define DIMS 1000
#define NUM_SV 1000
#define DIST 0.5

float64_t* lab;
float32_t* feat;
float64_t* alphas;
int32_t* svs;

void gen_rand_data()
{
	lab=SG_MALLOC(float64_t, NUM);
	feat=SG_MALLOC(float32_t, NUM*DIMS);
	alphas=SG_MALLOC(float64_t, NUM_SV);
	svs=SG_MALLOC(int32_t, NUM_SV);
	for (int32_t i=0; i<NUM; i++)
	{
		if (i<NUM/2)
		{
			lab[i]=-1.0;

			for (int32_t j=0; j<DIMS; j++)
				feat[i*DIMS+j]=CMath::random(0.0,1.0)+DIST;
		}
		else
		{
			lab[i]=1.0;

			for (int32_t j=0; j<DIMS; j++)
				feat[i*DIMS+j]=CMath::random(0.0,1.0)-DIST;
		}
	}
	for(int32_t i=0; i<NUM_SV; i++){
	  alphas[i] = 1/*CMath::random(0.0,1.0)*/;
	  svs[i] = i;
	}
}

int main()
{

	const int32_t feature_cache=0;
	const int32_t kernel_cache=0;
	const float64_t rbf_width=10;
	const float64_t svm_C=10;
	const float64_t svm_eps=0.001;
	
	std::cout.precision(30);


	Timer t;
	init_shogun();

	gen_rand_data();

	// create train labels
	CLabels* labels=new CLabels(SGVector<float64_t>(lab, NUM));
	SG_REF(labels);
	
	// create train features
	CSimpleFeatures<float32_t>* features = new CSimpleFeatures<float32_t>(feature_cache);
	SG_REF(features);
	features->set_feature_matrix(feat, DIMS, NUM);

	// create gaussian kernel
	CGaussianKernel* kernel = new CGaussianKernel(kernel_cache, rbf_width);
	SG_REF(kernel);
	kernel->init(features, features);

	// create svm via libsvm and train
	CLibSVM* svm = new CLibSVM(svm_C, kernel, labels);
	SG_REF(svm);
	svm->set_epsilon(svm_eps);
// 	svm->train();
	
	//Manually set support_vectors and Alphas
	svm->set_alphas(SGVector<float64_t>(alphas,NUM_SV));
	svm->set_support_vectors(SGVector<int32_t>(svs,NUM_SV));	
	
	std::cout << "Benchmarking cpu..." << std::endl;
	t.start();
	CLabels* out_labels=svm->apply();
	std::cout << "CPU Apply time : " << t.get() << std::endl;
	

	std::cout << "Benchmarking OpenCL Device..." << std::endl;
	std::cout << viennacl::ocl::current_device().info() << std::endl;
	std::cout << " Support double : " << viennacl::ocl::current_device().double_support() << std::endl;
	t.start();
	CLabels* ocl_out_labels=svm->ocl_apply();
	std::cout << "OpenCL Apply time : " << t.get() << std::endl;
	

	double total_diff = 0;

	for (int32_t i=0; i<NUM; i++){
		double diff =  CMath::abs((double)out_labels->get_label(i) - (double)ocl_out_labels->get_label(i));
// 		std::cout << "CPU " << out_labels->get_label(i) << " | GPU " << (double)ocl_out_labels->get_label(i) << std::endl;
// 		if(diff>0)
// 		  std::cout << "Difference between GPU and CPU at pos " << i << " " << diff <<  << std::endl;
		total_diff += diff;
	}

	std::cout << "Total  error : " << total_diff << std::endl;
	
	SG_UNREF(labels);
	SG_UNREF(out_labels);
	SG_UNREF(kernel);
	SG_UNREF(features);
	SG_UNREF(svm);

	exit_shogun();
	return 0;
}
