#include "kernel/GaussianKernel.h"
#include "features/RealFeatures.h"
#include "classifier/svm/LibSVM.h"
#include "lib/Mathematics.h"

#include <stdlib.h>
#include <stdio.h>

#define NUM 100
#define DIMS 2
#define DIST 0.5

double* lab;
double* feat;

void gen_rand_data()
{
	lab=new double[NUM];
	feat=new double[NUM*DIMS];

	for (int i=0; i<NUM; i++)
	{
		if (i<NUM/2)
		{
			lab[i]=-1.0;

			for (int j=0; j<DIMS; j++)
				feat[i*DIMS+j]=CMath::random(0.0,1.0)+DIST;
		}
		else
		{
			lab[i]=1.0;

			for (int j=0; j<DIMS; j++)
				feat[i*DIMS+j]=CMath::random(0.0,1.0)-DIST;
		}
	}
	CMath::display_vector(lab,NUM);
	CMath::display_matrix(feat,DIMS, NUM);
}

int main()
{

	const int feature_cache=0;
	const int kernel_cache=0;
	const double rbf_width=10;
	const double svm_C=10;
	const double svm_eps=0.001;

	gen_rand_data();

	CLabels labels;
	labels.set_labels(lab, NUM);

	CRealFeatures features(feature_cache);
	features.set_feature_matrix(feat, DIMS, NUM);

	CGaussianKernel kernel(kernel_cache, rbf_width);
	kernel.init(&features, &features);

	CLibSVM svm(svm_C, &kernel, &labels);
	svm.set_epsilon(svm_eps);
	svm.train();

	printf("num_sv:%d b:%f\n", svm.get_num_support_vectors(), svm.get_bias());

	CLabels out_labels(NUM);
	svm.classify(&out_labels);

	for (int i=0; i<NUM; i++)
		printf("out[%d]=%f\n", i, out_labels.get_label(i));

	return 0;
}
