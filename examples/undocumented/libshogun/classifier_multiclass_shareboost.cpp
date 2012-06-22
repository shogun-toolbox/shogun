#include <algorithm>

#include <shogun/labels/MulticlassLabels.h>
#include <shogun/io/StreamingAsciiFile.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/StreamingDenseFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/base/init.h>
#include <shogun/multiclass/ShareBoost.h>

#define  EPSILON  1e-5

using namespace shogun;

int main(int argc, char** argv)
{
	int32_t num_vectors = 0;
	int32_t num_feats   = 0;

	init_shogun_with_defaults();

	const char*fname_train = "../data/7class_example4_train.dense";
	CStreamingAsciiFile *train_file = new CStreamingAsciiFile(fname_train);
	SG_REF(train_file);

	CStreamingDenseFeatures<float64_t> *stream_features = new CStreamingDenseFeatures<float64_t>(train_file, true, 1024);
	SG_REF(stream_features);

	SGMatrix<float64_t> mat;
	SGVector<float64_t> labvec(1000);

	stream_features->start_parser();
	SGVector< float64_t > vec;
	while (stream_features->get_next_example())
	{
		vec = stream_features->get_vector();
		if (num_feats == 0)
		{
			num_feats = vec.vlen;
			mat = SGMatrix<float64_t>(num_feats, 1000);
		}
		std::copy(vec.vector, vec.vector+vec.vlen, mat.get_column_vector(num_vectors));
		labvec[num_vectors] = stream_features->get_label();
		num_vectors++;
		stream_features->release_example();
	}
	stream_features->end_parser();
	mat.num_cols = num_vectors;
	labvec.vlen = num_vectors;
	
	CMulticlassLabels* labels = new CMulticlassLabels(labvec);
	SG_REF(labels);

	// Create features with the useful values from mat
	CDenseFeatures< float64_t >* features = new CDenseFeatures<float64_t>(mat);
	SG_REF(features);

	SG_SPRINT("Performing ShareBoost on a %d-class problem\n", labels->get_num_classes());

	// Create ShareBoost Machine
	CShareBoost *machine = new CShareBoost(features, labels, 10);
	SG_REF(machine);

	machine->train();

	SGVector<int32_t> activeset = machine->get_activeset();
	SG_SPRINT("%d out of %d features are selected:\n", activeset.vlen, mat.num_rows);
	for (int32_t i=0; i < activeset.vlen; ++i)
		SG_SPRINT("activeset[%02d] = %d\n", i, activeset[i]);

	// Classify the training examples and show the results
	//CMulticlassLabels* output = CMulticlassLabels::obtain_from_generic(mc_svm->apply());

	//SGVector< int32_t > out_labels = output->get_int_labels();
	//SGVector<int32_t>::display_vector(out_labels.vector, out_labels.vlen);

	// Free resources
	SG_UNREF(machine);
	//SG_UNREF(output);
	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(train_file);
	SG_UNREF(stream_features);
	exit_shogun();

	return 0;
}

