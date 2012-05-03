#include <shogun/features/Labels.h>
#include <shogun/io/StreamingAsciiFile.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/StreamingSimpleFeatures.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/multiclass/MulticlassOneVsOneStrategy.h>
#include <shogun/machine/LinearMulticlassMachine.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/base/init.h>

#define  EPSILON  1e-5

using namespace shogun;

int main(int argc, char** argv)
{
	int32_t num_vectors = 0;
	int32_t num_feats   = 2;

	init_shogun_with_defaults();

	// Prepare to read a file for the training data
	char fname_feats[]  = "../data/fm_train_real.dat";
	char fname_labels[] = "../data/label_train_multiclass.dat";
	CStreamingAsciiFile* ffeats_train  = new CStreamingAsciiFile(fname_feats);
	CStreamingAsciiFile* flabels_train = new CStreamingAsciiFile(fname_labels);
	SG_REF(ffeats_train);
	SG_REF(flabels_train);

	CStreamingSimpleFeatures< float64_t >* stream_features = 
		new CStreamingSimpleFeatures< float64_t >(ffeats_train, false, 1024);

	CStreamingSimpleFeatures< float64_t >* stream_labels = 
		new CStreamingSimpleFeatures< float64_t >(flabels_train, true, 1024);

	SG_REF(stream_features);
	SG_REF(stream_labels);

	// Create a matrix with enough space to read all the feature vectors
	SGMatrix< float64_t > mat = SGMatrix< float64_t >(num_feats, 1000);

	// Read the values from the file and store them in mat
	SGVector< float64_t > vec;
	stream_features->start_parser();
	while ( stream_features->get_next_example() )
	{
		vec = stream_features->get_vector();

		for ( int32_t i = 0 ; i < num_feats ; ++i )
			mat[num_vectors*num_feats + i] = vec[i];

		num_vectors++;
		stream_features->release_example();
	}
	stream_features->end_parser();

	// Create features with the useful values from mat
	CSimpleFeatures< float64_t >* features = new CSimpleFeatures< float64_t >(mat.matrix, num_feats, num_vectors);

	CLabels* labels = new CLabels(num_vectors);
	SG_REF(features);
	SG_REF(labels);

	// Read the labels from the file
	int32_t idx = 0;
	stream_labels->start_parser();
	while ( stream_labels->get_next_example() )
	{
		labels->set_int_label( idx++, (int32_t)stream_labels->get_label() );
		stream_labels->release_example();
	}
	stream_labels->end_parser();

	// Create liblinear svm classifier with L2-regularized L2-loss
	CLibLinear* svm = new CLibLinear(L2R_L2LOSS_SVC);
	SG_REF(svm);

	// Add some configuration to the svm
	svm->set_epsilon(EPSILON);
	svm->set_bias_enabled(true);

	// Create a multiclass svm classifier that consists of several of the previous one
	CLinearMulticlassMachine* mc_svm = new CLinearMulticlassMachine(
			new CMulticlassOneVsOneStrategy(), (CDotFeatures*) features, svm, labels);
	SG_REF(mc_svm);

	// Train the multiclass machine using the data passed in the constructor
	mc_svm->train();

	// Classify the training examples and show the results
	CLabels* output = mc_svm->apply();

	SGVector< int32_t > out_labels = output->get_int_labels();
	CMath::display_vector(out_labels.vector, out_labels.vlen);

	// Free resources
	SG_UNREF(mc_svm);
	SG_UNREF(svm);
	SG_UNREF(output);
	SG_UNREF(features);
	SG_UNREF(labels);
	//SG_UNREF(ffeats_train);
	//SG_UNREF(flabels_train);
	SG_UNREF(stream_features);
	SG_UNREF(stream_labels);
	exit_shogun();

	return 0;
}
