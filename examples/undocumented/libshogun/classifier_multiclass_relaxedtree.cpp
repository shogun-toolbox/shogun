#include <algorithm>

#include <shogun/labels/MulticlassLabels.h>
#include <shogun/io/streaming/StreamingAsciiFile.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DenseSubsetFeatures.h>
#include <shogun/base/init.h>
#include <shogun/multiclass/tree/RelaxedTree.h>
#include <shogun/multiclass/MulticlassLibLinear.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/kernel/GaussianKernel.h>

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
	int32_t num_vec=0;
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
		num_vec++;

		if (num_vec > 20000)
			break;
	}
	stream_features->end_parser();
	mat.num_cols = num_vectors;
	labvec.vlen = num_vectors;
	
	CMulticlassLabels* labels = new CMulticlassLabels(labvec);
	SG_REF(labels);

	// Create features with the useful values from mat
	CDenseFeatures< float64_t >* features = new CDenseFeatures<float64_t>(mat);
	SG_REF(features);

	// Create RelaxedTree Machine
	CRelaxedTree *machine = new CRelaxedTree();
	SG_REF(machine);
	machine->set_labels(labels);
	CKernel *kernel = new CGaussianKernel();
	SG_REF(kernel);
	machine->set_kernel(kernel);

	CMulticlassLibLinear *svm = new CMulticlassLibLinear();

	machine->set_machine_for_confusion_matrix(svm);
	machine->train(features);

	
	CMulticlassLabels* output = CMulticlassLabels::obtain_from_generic(machine->apply());

	CMulticlassAccuracy *evaluator = new CMulticlassAccuracy();
	SG_SPRINT("Accuracy = %.4f\n", evaluator->evaluate(output, labels));

	// Free resources
	SG_UNREF(machine);
	SG_UNREF(output);
	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(train_file);
	SG_UNREF(stream_features);
	SG_UNREF(evaluator);
	SG_UNREF(kernel);

	exit_shogun();

	return 0;
}

