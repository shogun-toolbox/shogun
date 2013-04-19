#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <iostream>
using namespace shogun;
void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}
//generates data points (of different classes) randomly
void gen_rand_data(SGMatrix<float64_t> features, SGVector<float64_t> labels, float64_t distance)
{
	index_t num_samples=labels.vlen;
	index_t dimensions=features.num_rows;
	for (int32_t i=0; i<num_samples; i++) 
	{
		if (i<num_samples/2)	
		{
			labels[i]=-1.0;
			for(int32_t j=0; j<dimensions; j++)
				features(j,i)=CMath::random(0.0,1.0)+distance;
		}
		else
		{
			labels[i]=1.0;
			for(int32_t j=0; j<dimensions; j++)
				features(j,i)=CMath::random(0.0,1.0)-distance;
		}
	}
	labels.display_vector("labels");
	features.display_matrix("features");
}
int main(int argc, char** argv)
{
	init_shogun(&print_message, &print_message, &print_message);
	
	const int32_t feature_cache=0;
	const float64_t svm_C=10;
	const float64_t svm_eps=0.001;

	index_t num_samples=20;
	index_t dimensions=2;
	float64_t dist=0.5;

	SGMatrix<float64_t> featureMatrix(dimensions,num_samples);
	SGVector<float64_t> labelVector(num_samples);
	//random generation of data
	gen_rand_data(featureMatrix,labelVector,dist);
	
	//create train labels
	CLabels* labels=new CBinaryLabels(labelVector);
	
	//create train features
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(0);
	SG_REF(features);
	features->set_feature_matrix(featureMatrix); 
	
	//create linear kernel
	CLinearKernel* kernel=new CLinearKernel();
	SG_REF(kernel);
	kernel->init(features, features);
	
	//create svm classifier by LibSVM
	CLibSVM* svm=new CLibSVM(svm_C,kernel, labels);
	SG_REF(svm);
	svm->set_epsilon(svm_eps);
	svm->train();
	
	//classify data points 
	CBinaryLabels* out_labels=CBinaryLabels::obtain_from_generic(svm->apply());
	
	//convert scores to calibrated probabilities  by fitting a sigmoid function 
    //using the method described in Lin, H., Lin, C., and Weng,  R. (2007). A note 
	//on Platt's probabilistic outputs for support vector machines.	
	out_labels->scores_to_probabilities();
	
	//display output labels and probabilities
	for (int32_t i=0; i<num_samples; i++)
	{
		SG_SPRINT("out[%d]=%f (%f)\n", i, out_labels->get_label(i),
				out_labels->get_value(i));
	}

	//clean up	
	SG_UNREF(out_labels);
	SG_UNREF(kernel);
	SG_UNREF(features);
	SG_UNREF(svm);	
	
	exit_shogun();	
	
	return 0;
}
