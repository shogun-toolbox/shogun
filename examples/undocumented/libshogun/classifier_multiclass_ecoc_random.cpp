#include <shogun/io/CSVFile.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/multiclass/ecoc/ECOCStrategy.h>
#include <shogun/multiclass/ecoc/ECOCRandomDenseEncoder.h>
#include <shogun/multiclass/ecoc/ECOCRandomSparseEncoder.h>
#include <shogun/multiclass/ecoc/ECOCHDDecoder.h>
#include <shogun/machine/LinearMulticlassMachine.h>
#include <shogun/classifier/svm/LibLinear.h>

#define  EPSILON  1e-5

using namespace shogun;

/* file data */
const char fname_feats[]="../data/fm_train_real.dat";
const char fname_labels[]="../data/label_train_multiclass.dat";

void test()
{
	/* dense features from matrix */
	CSVFile* feature_file = new CSVFile(fname_feats);
	SGMatrix<float64_t> mat=SGMatrix<float64_t>();
	mat.load(feature_file);

	DenseFeatures<float64_t>* features=new DenseFeatures<float64_t>(mat);

	/* labels from vector */
	CSVFile* label_file = new CSVFile(fname_labels);
	SGVector<float64_t> label_vec;
	label_vec.load(label_file);

	MulticlassLabels* labels=new MulticlassLabels(label_vec);

	// Create liblinear svm classifier with L2-regularized L2-loss
	LibLinear* svm = new LibLinear(L2R_L2LOSS_SVC);

	// Add some configuration to the svm
	svm->set_epsilon(EPSILON);
	svm->set_bias_enabled(true);

	// Create a multiclass svm classifier that consists of several of the previous one
	LinearMulticlassMachine* mc_svm = new LinearMulticlassMachine(
			new CECOCStrategy(new CECOCRandomDenseEncoder(), new CECOCHDDecoder()), (DotFeatures*) features, svm, labels);

	// Train the multiclass machine using the data passed in the constructor
	mc_svm->train();

	// Classify the training examples and show the results
	MulticlassLabels* output = mc_svm->apply()->as<MulticlassLabels>();

	SGVector< int32_t > out_labels = output->get_int_labels();
	SGVector< int32_t >::display_vector(out_labels.vector, out_labels.vlen);

	// Free resources
}

int main(int argc, char** argv)
{
//	env()->io()->set_loglevel(MSG_DEBUG);

	test();

	return 0;
}

