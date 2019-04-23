#include <iostream>
#include <shogun/base/init.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/evaluation/SigmoidCalibration.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

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
                features(j,i)=Math::random(0.0,1.0)+distance;
        }
        else
        {
            labels[i]=1.0;
            for(int32_t j=0; j<dimensions; j++)
                features(j,i)=Math::random(0.0,1.0)-distance;
        }
    }
    labels.display_vector("labels");
    std::cout<<std::endl;
    features.display_matrix("features");
    std::cout<<std::endl;
}

int main(int argc, char** argv)
{
    init_shogun_with_defaults();

    const float64_t svm_C=10;

    index_t num_samples=20;
    index_t dimensions=2;
    float64_t dist=0.5;

    SGMatrix<float64_t> featureMatrix(dimensions,num_samples);
    SGVector<float64_t> labelVector(num_samples);
    //random generation of data
    gen_rand_data(featureMatrix,labelVector,dist);

    //create train labels
    auto labels=std::make_shared<BinaryLabels>(labelVector);

	// create train features
	auto features = std::make_shared<DenseFeatures<float64_t>>(featureMatrix);

	// create linear kernel
	auto kernel = std::make_shared<LinearKernel>();
	kernel->init(features, features);

	// create svm classifier by LibSVM
	auto svm = std::make_shared<LibSVM>(svm_C, kernel, labels);
	svm->train();

	// classify data points
	auto out_labels = svm->apply()->as<BinaryLabels>();

	/*convert scores to calibrated probabilities  by fitting a sigmoid function
	using the method described in Lin, H., Lin, C., and Weng,  R. (2007). A note
	on Platt's probabilistic outputs for support vector machines.
	See BinaryLabels documentation for details*/
	auto sigmoid_calibration = std::make_shared<SigmoidCalibration>();
	sigmoid_calibration->fit_binary(
	    out_labels, labels->as<BinaryLabels>());
	auto calibrated_labels =
	    sigmoid_calibration->calibrate_binary(out_labels);

	// display output labels and probabilities
	for (int32_t i = 0; i < num_samples; i++)
	{
		SG_SPRINT(
		    "out[%d]=%f (%f)\n", i, out_labels->get_label(i),
		    calibrated_labels->get_value(i));
	}

	// clean up

	exit_shogun();

	return 0;
}
